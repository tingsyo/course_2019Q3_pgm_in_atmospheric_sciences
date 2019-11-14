#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The original VVM surface precipitation data was stored in one file of (n, y, x). 
This script splited the data into n files of (y,x).
"""
import os, csv, logging, argparse
import numpy as np
import datetime

# Set up a fake starting time for file-name: 2001-01-01-00
t0 = dateimte.datetime.strptime('2001010100', '%Y%m%d%H')

# For 8x8 grid average
for i in range(tmp.shape[0]):
    t = t0 + datetime.timedelta(hours=i)
    fname = datetime.datetime.strftime(t, '%Y%m%d%H%M')+'.npy'
    np.save('scale_0.125/'+fname, tmp[i,:,:])

# For 4x4 average
tmp = np.load('X.256.sprec.npy')
for i in range(tmp.shape[0]):
    t = t0 + datetime.timedelta(hours=i)
    fname = datetime.datetime.strftime(t, '%Y%m%d%H%M')+'.npy'
    np.save('scale_0.25/'+fname, tmp[i,:,:])

# For 2x2 average
tmp = np.load('X.512.sprec.npy')
for i in range(tmp.shape[0]):
    t = t0 + datetime.timedelta(hours=i)
    fname = datetime.datetime.strftime(t, '%Y%m%d%H%M')+'.npy'
    np.save('scale_0.5/'+fname, tmp[i,:,:])

# For original grid scale
tmp = np.load('Y.1024.sprec.npy')
for i in range(tmp.shape[0]):
    t = t0 + datetime.timedelta(hours=i)
    fname = datetime.datetime.strftime(t, '%Y%m%d%H%M')+'.npy'
    np.save('original/'+fname, tmp[i,:,:])
