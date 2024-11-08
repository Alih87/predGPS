#!/usr/bin/env python
from collections import deque
from math import atan, atan2, pi
from numpy import random, array, linalg, matmul, zeros, eye

class KF(object):
	def __init__(self):
		self.priori_est, self.post_est, self.P = array(([[0]]),ndmin=2), array(([[0]]),ndmin=2), array(([[1]]))
		self.MAX_LEN = 1
		
		self.moving_avg = deque(maxlen=self.MAX_LEN)
		self.A, self.B = array(([[1]]), ndmin=2), array(([[1]]), ndmin=2)

	def get_initial_values(self):
		self.initial_heading = self.theta_mag
		self.prev_theta_mag = self.initial_heading
		self.moving_avg.extend([self.initial_heading]*self.MAX_LEN)
		self.offset = self.theta_mag - self.theta_yaw
		self.prev_theta_scout = self.theta_scout

	def get_priori_est(self):
		self.priori_est = matmul(self.A, array(([[self.theta_scout + self.initial_heading]]), ndmin=2)) + matmul(self.B, array(([[self.delta_theta]]), ndmin=2))
		self.priori_est = (self.priori_est + 180) % 360 - 180
	
	def get_measurements(self):
		self.mag_measure, self.yaw_measure = self.theta_mag, self.theta_yaw + self.offset
		self.mag_measure, self.yaw_measure = (self.mag_measure + 180) % 360 - 180, (self.yaw_measure + 180) % 360 - 180

	def get_posteriori_est(self):
		Q = random.normal(0, 0.7, 1)[0]
		C = array(([[1],[1]]), ndmin=2)
		R = array(([[random.normal(0, 1.959, 1)[0], 0],
			    [0, 0.001]]))
		last_K_ = array(([[0.5, 0.5]]), ndmin=2)

		# Get priori estimate and predicted error covariance
		self.get_priori_est()
		P_ = matmul(self.A, matmul(self.P, self.A.T)) + Q

		# Update using measurements
		K_ = matmul(matmul(P_, C.T), linalg.inv(matmul(C, matmul(P_, C.T)) + R))
		self.get_measurements()
		y = array(([[self.mag_measure],[self.yaw_measure]]), ndmin=2)
		if self.priori_est - matmul(K_,(y - matmul(C, self.priori_est))) > 185 or self.priori_est - matmul(K_,(y - matmul(C, self.priori_est))) < -185:
			self.post_est += self.delta_theta
		else:	
			self.post_est = self.priori_est - matmul(K_,(y - matmul(C, self.priori_est)))
		self.post_est = (self.post_est + 180) % 360 - 180
		self.P = matmul((array([[1]],ndmin=2) - matmul(K_, C)), P_)
		#self.priori_est = self.post_est
