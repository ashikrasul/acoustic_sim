#!/usr/bin/env python

from datetime import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from .UKF_class import UKF
from .EKF_class import EKF
import rospy
from .process_model_class import ProcessModelVelecitiesGlobal
from .meas_model_class import MeasurementModelDistances
from collections import deque
import threading
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import FluidPressure
import os
import json

class localisationSimulation():
    def __init__(self):

        # Load configuration files relative to this Python module so the class
        # works both from the source tree and from an installed ROS package.
        tmp = os.path.dirname(__file__)
        file_path_filter = os.path.join(tmp, '../../config/acoustic_config.json')
        f = open(file_path_filter)
        self.acoustic_config = json.load(f)
        f.close()

        file_path_filter = os.path.join(tmp, '../../config/filter_config.json')
        f = open(file_path_filter)
        self.filter_config = json.load(f)
        f.close()
        
        
        
        self.t0 = 0
        self.dt = 0
        self.t = self.t0
        self.last_t = self.t0
        self.z = None
        self.x = None
        self.x_est = None
        self.statePre = None
        self.covarPre = None
        self.p_mat = None 
    
        # Read acoustic timing settings and select the configured Kalman filter.
        self.packetLengthResponse = self.acoustic_config["config"][0]["PacketLengthResponse"]
        self.publishDelay = self.acoustic_config["config"][0]["PublishDelay"]
        self.filter = self.filter_config["config"][0]["filterTyp"]
        if self.filter == "UKF":
            self.x0 = np.array(self.filter_config["config"][1]["settings"]["InitState"])
            self.p_mat_0 = np.array(self.filter_config["config"][1]["settings"]["InitCovar"])
        elif self.filter == "EKF":
            self.x0 = np.array(self.filter_config["config"][2]["settings"]["InitState"])
            self.p_mat_0 = np.array(self.filter_config["config"][2]["settings"]["InitCovar"])

        else: print("[Localisation_sim] Wrong Filter selected")
        # Measurement-noise values used by the depth and range updates.
        self.w_mat_depth = self.filter_config["config"][2]["settings"]["Qt_depth"]
        self.w_mat_dist = self.filter_config["config"][2]["settings"]["Qt_dist"]
        
        # Keep recent filter snapshots so delayed acoustic measurements can be
        # applied at their original time and the newer history can be replayed.
        self.dataBag = deque([])
        self.lenDataBag = self.filter_config["config"][0]["lengthDatabag"]
        
        # These lists are reserved for storing simulation results.
        self.xest = []
        self.yest = []
        self.zest = []
        self.timeest = []

        # The state is [x, y, z]. The process model integrates global velocity,
        # while the measurement model converts position into range/depth values.
        self.measurement_model = MeasurementModelDistances(1,1,1,1)
        self.process_model = ProcessModelVelecitiesGlobal(3) # 3 = dim_state
        if self.filter == "UKF":
            self.Kalmanfilter = UKF(self.measurement_model, self.process_model, self.x0, self.p_mat_0)  # prediction and update done in this instance

        elif self.filter == "EKF":
            self.Kalmanfilter = EKF(self.measurement_model, self.process_model, self.x0, self.p_mat_0)  # prediction and update done in this instance
     

    def fillDatabag(self, list):
        # Each entry is [time, velocity_input, depth, state_estimate, covariance].
        # The oldest snapshot is discarded once the configured history length
        # is exceeded.
        self.dataBag.append(list)
        if len(self.dataBag)> self.lenDataBag:
            self.dataBag.popleft()

    def recalculateState(self, correctedTime, measurements):
        # Acoustic ranges are published after the physical measurement time.
        # Find the newest saved snapshot at or before correctedTime by walking
        # backward through the deque.
        numberIterations = 0
        for i in range(len(self.dataBag)):
            if correctedTime >= self.dataBag[-i-1][0]:
                break
            elif correctedTime < self.dataBag[-i-1][0]:
                numberIterations +=1
                if numberIterations == len(self.dataBag):
                    numberIterations = 0 # CorrectedTime is newer than every safed time
                    break
            else:
                print("Error: [Localisation_Sim]; no matching timestamp found") 

        # Restore the historical state and advance it to the range measurement.
        # The saved entry layout is [time, velocity, depth, state, covariance].
        self.setFilter(self.dataBag[-numberIterations-1][3], self.dataBag[-numberIterations-1][4], self.dataBag[-numberIterations-1][0])
        self.update(correctedTime, self.dataBag[-numberIterations-1][1],measurements)
        
        # Replay all inputs that occurred after the delayed measurement so the
        # filter ends at the current simulation time rather than in the past.
        for i in range(numberIterations):
            self.xest = self.predict(self.dataBag[-numberIterations+i][0], self.dataBag[-numberIterations+i][1], self.dataBag[-numberIterations+i][2])

    def setFilter (self, x_est, p_mat, t):
        # Restore all three parts of the filter's temporal state before replay.
        self.Kalmanfilter.set_state(x_est)
        self.Kalmanfilter.set_covar(p_mat)
        self.Kalmanfilter.set_time(t)
        
    def update(self, correctedTime, preInput, measurements):
        # First predict from the restored snapshot to the measurement time,
        # then correct the position using the beacon range.
        self.Kalmanfilter.predict(correctedTime, preInput)  # launch prediction step with time stamp and noisy velocity; 
        self.x_est, self.p_mat, z = self.Kalmanfilter.update_dist(measurements, self.w_mat_dist) # launch update step with published data; return: self.x_est = updated state, z = delta between z and zhat       

    def predict(self, t, preInput, depth): # just a function for debugging and to have a camparison
        # Propagate position using the supplied global velocity and elapsed time.
        self.x_est, self.p_mat = self.Kalmanfilter.predict(t, preInput)  # launch prediction step with time stamp and noisy velocity; return: x = predicted state, p = predicted covariance
        # Depth is treated as a direct measurement of the z component.
        self.x_est, self.p_mat = self.Kalmanfilter.update_depth(depth, self.w_mat_depth)
            

    def locate(self, preInput, t, depth, meas):
        # This method is called once per simulation/update tick.
        self.t = t
        self.dt = self.t-self.last_t
        self.last_t = self.t
                
        # If a delayed acoustic range is available, rewind and replay history.
        if meas is not None:
            # Expected measurement fields include beacon position, measured
            # distance, publication time, and response-packet duration.
            correctedTime = meas["time_published"] - meas["PacketLengthResponse"]  # get time stamp
            beacon = meas["ModemPos"]
            dist = meas["dist"]
            measurements = [beacon, dist] # Position Beacon, Distance
            self.recalculateState(correctedTime, measurements)

        else:
            # No range update this tick: perform the normal prediction and
            # depth correction using the current time and vehicle inputs.
            self.predict(self.t, preInput, depth)
        
        list = [self.t, preInput, depth, self.x_est, self.p_mat]
        self.fillDatabag(list)
        return self.x_est
    
    def getBeaconPos(self, BeaconIndex):
        # Look up an anchor's fixed position by its modem ID.
        for i in self.acoustic_config["config"]:
            if i["type"] == "anchor":
                if i["modem"]["id"] == BeaconIndex:
                    return i["position"]