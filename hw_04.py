import numpy as np
import cv2 
import copy

num = 5000 #Number of samples

u_sig = 0.5
v_sig = 0.5
r_sig = 0.1

logName = "data/log/robotdata1.log"

gridMap = np.loadtxt("data/map/wean.dat", skiprows = 7)

count_sample = 0

is_resample = 0

samples = np.zeros((num, 3)) # (x, y, th)
sigmas = np.zeros((num, 3)) #!!!! Each should be a 3x3 matrix
weights = np.ones(num)

in_i = []
in_j = []
count_in = 0

for i in range(0, 800):
    for j in range(0, 800):
        if gridMap[i, j] == 1:
            in_i.append(i)
            in_j.append(j)
            count_in = count_in + 1

while count_sample < num:
    seed = np.random.uniform(0, count_in, 1)
    samples[count_sample, 0] = in_i[seed.astype(int)]
    samples[count_sample, 1] = in_j[seed.astype(int)]
    theta = np.random.uniform(0, np.pi*2., 1)
    samples[count_sample, 2] = theta
    count_sample = count_sample + 1


tryMap = copy.copy(gridMap)
for s in samples:
    tryMap[s[0], s[1]] = 0.2


logFile = open(logName)
last_odom = np.zeros(3)

t = 0
d_th = 0
d_u = 0
d_v = 0

for line in logFile:
    elements = line.split()
    if elements[0] == "L": #!!!! Only use odom from L lines...
       odom = [float(e) for e in elements[4:7]] #!!!! odom of laser, not robot
       laser = [float(e) for e in elements[7:187]]
       if t != 0: #most cases except first
           d_th = odom[2] - last_odom[2]
           d_x = odom[0] - last_odom[0]
           d_y = odom[1] - last_odom[1]
           d_u = (d_x*np.cos(-odom[2]) - d_y*np.sin(-odom[2]))/10.
           d_v = (d_x*np.sin(-odom[2]) + d_y*np.cos(-odom[2]))/10.
      
       valid_num = 0
       for i in range(0, num):
           s = samples[i, :]
           sig = sigmas[i, :]
           d_x = d_u*np.cos(s[2]) - d_v*np.sin(s[2])
           d_y = d_u*np.sin(s[2]) + d_v*np.cos(s[2])
           samples[i, 0] = s[0] + d_x
           samples[i, 1] = s[1] + d_y
           samples[i, 2] = s[2] + d_th
           sigmas[i, 0] = sig[0] + u_sig
           sigmas[i, 1] = sig[1] + v_sig
           sigmas[i, 2] = sig[2] + r_sig


           if (samples[i, 0] < 0) or (samples[i, 0] >= 800) or (samples[i, 1] < 0) or (samples[i, 1] >= 800):
               weights[i] = 0
           elif gridMap[samples[i, 0], samples[i, 1]] != 1:
               weights[i] = 0
           elif weights[i] < 0.01/num:
               weights[i] = 0
           else:
               valid_num = valid_num + 1

       weights = weights/valid_num
       if valid_num < 0.95*num:
           is_resample = 1
           print valid_num

#TODO PF sensor part
       idx = 0
       ds_rate = 10
       for s, sig, w in zip(samples, sigmas, weights):
           if w != 0:
               l_x = np.zeros(180/ds_rate)
               l_y = np.zeros(180/ds_rate)
               rad = -np.pi/2. + np.pi/360.
               for j in range(0, 180/ds_rate):
                   d = laser[j]
                   l_x[j] = (np.rint((d/10.)*np.cos(rad + s[2]) + s[0]))
                   l_y[j] = (np.rint((d/10.)*np.sin(rad + s[2]) + s[1]))
                   rad = rad + np.pi/180.*ds_rate

               count_match = 0
               for x, y in zip(l_x, l_y):
                   if (x >= 0) and (x < 800) and (y >= 0) and (y < 800):
                       if gridMap[x, y] == 0: #need better function()
                           count_match = count_match + 1

               weights[idx] = weights[idx] * count_match
           
           idx = idx + 1
       
       sum_w = np.sum(weights)
       weights = weights/sum_w
       print t

       if is_resample == 1:
           cdf = 0
           count_s = 0
           old_samples = copy.copy(samples)
           for s, sig, w in zip(old_samples, sigmas, weights):
               ratio = 1. + 10./(t + 10.)
               cdf = cdf + w*(num/ratio)
               while (count_s < cdf):
                   tmp_s = np.zeros((1, 3))
                   xy_seed = np.random.normal(0, sig[0], 2) #multivariate_normal
                   xy_noise = xy_seed.astype(int)
                   r_seed = np.random.normal(0, sig[2], 1) #multivariate_normal
                   r_noise = r_seed.astype(int)
                   tmp_s[0, 0] = s[0] + xy_noise[0]
                   tmp_s[0, 1] = s[1] + xy_noise[1]
                   tmp_s[0, 2] = s[2] + r_noise[0]
                   if (tmp_s[0, 0] >= 0) and (tmp_s[0, 0] < 800) and (tmp_s[0, 1] >= 0) and (tmp_s[0, 1] < 800):
                       if gridMap[tmp_s[0, 0], tmp_s[0, 1]] == 1:
                           samples[count_s, :] = tmp_s[0, :]
                           count_s = count_s + 1

               for i in range(count_s, num):
                   seed = np.random.uniform(0, count_in, 1)
                   samples[i, 0] = in_i[seed.astype(int)]
                   samples[i, 1] = in_j[seed.astype(int)]
                   theta = np.random.uniform(0, np.pi*2., 1)
                   samples[i, 2] = theta

       
           weights = np.ones(num)
           sigmas = np.zeros((num, 3))
           is_resample = 0
       
       last_odom = odom
       t = t + 1
       PFMap = copy.copy(gridMap)
       for s, w in zip(samples, weights):
           if w != 0:
               PFMap[s[0], s[1]] = 0.2

       cv2.imshow('PF', PFMap)
       cv2.waitKey(10)



    
