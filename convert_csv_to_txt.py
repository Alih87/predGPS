#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 07:10:27 2024

@author: hassan
"""

import pandas as pd
import numpy as np
import utm, math
import matplotlib.pyplot as plt

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z * (180/math.pi)

path = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS/"
csv_path_filtered = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS/gps-filtered.csv"
csv_path_raw = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS/ublox-fix.csv"
csv_ra = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS/imu-data_raw.csv"
csv_ne = "/home/hassan/Documents/scout/sixth_attempt_SUCCESS/imu-data.csv"

raw_utm, filtered_utm = [], []
raw_angles, new_angles = [], []

raw_csv = pd.read_csv(csv_path_raw)
filtered_csv = pd.read_csv(csv_path_filtered)
raw_ang = pd.read_csv(csv_ra)
new_ang = pd.read_csv(csv_ne)

f_lat, f_lon = filtered_csv['latitude'].to_list(), filtered_csv['longitude'].to_list()

plt.scatter(f_lat, f_lon)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Robot's trajectory")
plt.show()


r_lat, r_lon = raw_csv['latitude'].to_list(), raw_csv['longitude'].to_list()
xr, yr, zr, wr = raw_ang['orientation.x'].to_list(), raw_ang['orientation.y'].to_list(), raw_ang['orientation.z'].to_list(), raw_ang['orientation.w'].to_list()
xn, yn, zn, wn = new_ang['orientation.x'].to_list(), new_ang['orientation.y'].to_list(), new_ang['orientation.z'].to_list(), new_ang['orientation.w'].to_list()

plt.scatter(r_lat, r_lon)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Robot's trajectory")
plt.show()

raw_coords, filtered_coords = list(zip(r_lat, r_lon)), list(zip(f_lat, f_lon))

for p in raw_coords:
    lat, lon, _, _ = utm.from_latlon(p[0], p[1])
    raw_utm.append((lat, lon))
    
for p in filtered_coords:
    lat, lon, _, _ = utm.from_latlon(p[0], p[1])
    filtered_utm.append((lat, lon))
    
for i in range(len(xr)):
    raw_angles.append(euler_from_quaternion(xr[i], yr[i], zr[i], wr[i]))
    
for i in range(len(xn)):
    new_angles.append(euler_from_quaternion(xn[i], yn[i], zn[i], wn[i]))

with open(path+"raw_utm.txt", 'w') as f:
    for p in raw_utm:
        f.write(str(p[0])+","+str(p[1])+"\n")
    f.close()
    
with open(path+"filtered_utm.txt", 'w') as f:
    for p in filtered_utm:
        f.write(str(p[0])+","+str(p[1])+"\n")
    f.close()
    
with open(path+"raw_angles.txt", 'w') as f:
    for p in raw_angles:
        f.write(str(p)+"\n")
    f.close()
    
with open(path+"new_angles.txt", 'w') as f:
    for p in new_angles:
        f.write(str(p)+"\n")
    f.close()