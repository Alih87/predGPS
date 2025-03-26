#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 06:44:00 2024

@author: hassan
"""

import numpy as np
import pandas as pd
import bagpy as bp
from time import sleep
import utm
import math, copy
from collections import deque
 
def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z
    
def make_incremental(data):
    data1 = copy.copy(data)
    data1 = deque(data1)
    data1.appendleft(data[0])
    data1 = [*data1]
    
    final_data = []
    
    for idx in range(len(data)):
        final_data.append([data[idx][0]-data1[idx][0], data[idx][1]-data1[idx][1], data[idx][2], data[idx][3],
                            data[idx][4], data[idx][5], data[idx][6], data[idx][7], data[idx][8], data[idx][9],
                            data[idx][10]])
        # final_data.append([data[idx][0], data[idx][1], data[idx][2], data[idx][3],
        #                    data[idx][4], data[idx][5], data[idx][6], data[idx][7], data[idx][8], data[idx][9],
        #                    data[idx][10]])
    
    return final_data

bag_path = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS.bag"

bag = bp.bagreader(bag_path)

odom_gps = bag.message_by_topic('/gps/filtered')
raw_gps = bag.message_by_topic('/ublox/fix')

raw_angle = bag.message_by_topic('/imu/data_raw')
new_angle = bag.message_by_topic('/imu/data')
tf_tree = bag.message_by_topic('/tf')

sleep(0.25)

odom_gps_df = pd.read_csv(odom_gps)
raw_gps_df = pd.read_csv(raw_gps)
new_angle_df = pd.read_csv(new_angle)
tf_frames =  pd.read_csv(tf_tree)["transforms"].to_list()
count = 0

data = []

for sec_gps, nsec_gps in zip(new_angle_df["header.stamp.secs"].to_list(), new_angle_df["header.stamp.nsecs"].to_list()):    
    for sec_imu, nsec_imu in zip(odom_gps_df["header.stamp.secs"].to_list(), odom_gps_df["header.stamp.nsecs"].to_list()):
        if sec_gps == sec_imu and abs(int(nsec_imu) - int(nsec_gps) <= 500):
            for tf in tf_frames:
                if "imu_link" in tf.split("\n")[6] and "map" in tf.split("\n")[5]:
                    sec_odom = int(tf.split("\n")[3].split(": ")[1])
                    nsec_odom = int(tf.split("\n")[4].split(": ")[1]) 
                    if sec_odom == int(sec_gps) and abs(nsec_odom - int(nsec_imu) <= 400):
                        utm_x, utm_y, _, _ = utm.from_latlon(float(odom_gps_df["latitude"].iloc[count]),
                                                       float(odom_gps_df["longitude"].iloc[count]))
                        roll, pitch, yaw = euler_from_quaternion(float(tf.split("\n")[13].split(": ")[1]),
                                                                 float(tf.split("\n")[14].split(": ")[1]),
                                                                 float(tf.split("\n")[15].split(": ")[1]),
                                                                 float(tf.split("\n")[16].split(": ")[1][:-1]))
                        vx, vy, vz = new_angle_df["angular_velocity.x"].iloc[count], new_angle_df["angular_velocity.y"].iloc[count], new_angle_df["angular_velocity.z"].iloc[count]
                        ax, ay, az = new_angle_df["linear_acceleration.x"].iloc[count], new_angle_df["linear_acceleration.y"].iloc[count], new_angle_df["linear_acceleration.z"].iloc[count]
                        data.append((utm_x, utm_y, roll, pitch, yaw, vx, vy, vz, ax, ay, az))
                        break

    count += 1

final_data = make_incremental(data)

with open("/home/hassan/dataset_val.txt", 'w') as f:
    for idx, d in enumerate(final_data):
        if d[0] != 0.0 and d[1] != 0.0:
            f.write(str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3])+","+str(d[4])+","+str(d[5])+","+str(d[6])+","+str(d[7])+","+str(d[8])+","+str(d[9])+","+str(d[10])+"\n")
    f.close()
    