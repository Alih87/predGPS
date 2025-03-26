#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:59:06 2024

@author: hassan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_data(file_path):
    east, north, roll, pitch, yaw, vx, vy, vz, ax, ay, az = [], [], [], [], [], [], [], [], [], [], []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            try:
                data = line.strip().split(',')
                if len(data) >= 2:
                    east_val, north_val, roll_val, pitch_val, yaw_val, vx_val, vy_val, vz_val, ax_val, ay_val, az_val = map(float, data)
                    if east_val != 0.0 and north_val != 0.0:  # Exclude (0.0, 0.0)
                        east.append(east_val)
                        north.append(north_val)
                        roll.append(roll_val)
                        pitch.append(pitch_val)
                        yaw.append(yaw_val)
                        vx.append(vx_val)
                        vy.append(vy_val)
                        vz.append(vz_val)
                        ax.append(ax_val)
                        ay.append(ay_val)
                        az.append(az_val)

            except ValueError:
                continue
            
    return east, north, roll, pitch, yaw, vx, vy, vz, ax, ay, az
    
def read_gps_test(file_path):
    x, y = [], []
    with open(file_path, 'r') as f:
        r = f.readlines()
        f.close()
    for line in r:
        xx, yy = list(map(np.float64, line.split(",")))
        if xx != 0.0 and yy != 0.0:
            x.append(xx)
            y.append(yy)
    
    count= 0
    for i,j in zip(x,y):
        if i == 0.0 and j == 0.0:
            pass
        else:
            count += 1
    
    return x, y
    
def make_hist(x):
    vals_X, freq_X = np.unique(x, return_counts=True)
    plt.plot(vals_X, freq_X)
    plt.show()
    
if __name__ == '__main__':
    path = "/home/hassan/dataset_val.txt"
    
    x, y, roll, pitch, yaw, vx, vy, vz, ax, ay, az = parse_data(path)
    data_df = pd.DataFrame(data=[x, y, roll, pitch, yaw, vx, vy, vz, ax, ay, az],
                           index=["x","y","roll","pitch","yaw","vx","vy","vz","ax","ay","az"]).T
    
    east, north = [], []
    for idx, l in enumerate(zip(x, y)):
        if idx == 0:
            east.append(l[0])
            north.append(l[1])
        else:
            east.append(east[idx-1] + l[0])
            north.append(north[idx-1] + l[1])
    
    plt.plot(data_df["vz"])
    plt.title("vz")
    plt.show()
    
    plt.plot(data_df["yaw"])
    plt.title("yaw")
    plt.show()
    
    plt.plot(data_df["ax"])
    plt.title("ax")
    plt.show()
    
    print(np.std(data_df['ax']))
    
    plt.scatter(x, y)
    print("Mean Increment: ", str(np.mean(x)), str(np.mean(y)))
    print("Increment  std: ", str(np.std(x)), str(np.std(y)))
    plt.show()
    
    plt.plot(east, north)
    plt.title("Trajectory")
    plt.show()
    