# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:29:04 2019

@author: lenovo
"""

import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.io import loadmat

#read images and return them in a list each image is 381*481
def read_images(path):
    #list to hold images
    images = []
    #take onlt jpg files
    files = [file[:-4] for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file.endswith('jpg') ]
    files.sort(key=int)
    for file in files:
        file = file + '.jpg'
        images.append(img.imread(os.path.join(path, file))) 
    return images

#read he ground truth images for each image
def read_groundTruth(path):
    groundT = []
    #take only mat files
    files = [file[:-4] for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file.endswith('mat') ]
    files.sort(key=int)
    for file in files:
        file = file + '.mat'
        #load each .mat file for each image
        gt = loadmat(os.path.join(path,file))
        process_img = []
        #loop over the ground truth images of a real image 
        for k in np.squeeze(gt['groundTruth']).reshape(-1):
            process_img.append(k[0][0][0])

        groundT.append(process_img);
    return groundT
 
#display each image with its ground truth images
def display_img_with_groundT(images, gt, index):
    n = len(gt[index]) + 1
    fig, ax = plt.subplots(1, n, figsize=(20,20))
    ax[0].imshow(images[index])
    for i in range(len(gt[index])):
        ax[i+1].imshow(gt[index][i])
    plt.show()
        
    
    

images = read_images(path1)
gt = read_groundTruth(path2)
print(len(gt[0]))
for i in range(10):
    display_img_with_groundT(images, gt, i)