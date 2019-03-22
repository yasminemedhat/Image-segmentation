# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:24:33 2019

@author: Loujaina
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import img_segmentation as img
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import validation as validate
from PIL import Image   
import numpy as np


def display_img_with_segmented_img(images, filenames, clustered_imgs, index, k_list):
    #get number of images to be displayed; real image plus segmented image for each k in kmeans
    n = len(k_list)+1
    fig, ax = plt.subplots(1, n)
    fig.suptitle('image: ' + filenames[index])
    ax[0].imshow(images[index]) 
    for i in range(len(k_list)):
        ax[i+1].set_title('K = ' + str(k_list[i]));
        ax[i+1].imshow(clustered_imgs[index]);
    plt.show()   


def ncut_clustering(images, k):
    result = []
    i = 0
    for image in images:
        print('clustering image ',i)
        image2 = image.reshape(-1,3)/255 #divide by 255 to get values between 0-1->normalization
                                         #removes distortions
        
        s_clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',
                                          n_neighbors = 5,n_jobs=-1,
                                          eigen_solver='arpack').fit_predict(image2)
        
        result.append(s_clustering.reshape(image.shape[:-1]))
        #result.append(affinity_mat.reshape(image.shape[:-1]))
        i+=1
    return result

def resize_gt(g,im):
     groundT = []
     for i in range(15):
        soura=im[i]
        rows=soura.shape[0]
        cols=soura.shape[1]
        x=0;
        process_img = []
        for j in range(len(g[i])):

            copy=(g[i][j]).copy()
            new=np.resize(copy, (rows,cols))
            x+=1
            process_img.append(new);
        groundT.append(process_img)
     return groundT


def main():
    path_gt=r"D:\College\Term 8\Pattern Recognition\Project 2\bigpic\groundTruth"
    path_img=r"D:\College\Term 8\Pattern Recognition\Project 2\bigpic\test"
    path_resized=r"D:\College\Term 8\Pattern Recognition\Project 2\bigpic\resized"
    
    #load images and gt
    img_names, images = img.read_images(path_img)
    img_names2,resized=img.read_images(path_resized)
    gt = img.read_groundTruth(path_gt)
    
    
    #resize groundtruth to fit new dimensions
    groundT = resize_gt(gt,resized)
     
   
    
    kmeans_results = img.diff_kmeans(resized, [5])
    Fivenn = ncut_clustering(resized,[5])
    
      #contrast
    kmeans_total_fscore, kmeans_avg_fscore, kmeans_total_cEntropy, kmeans_avg_cEntropy = validate.validate_clustering(groundT, kmeans_results, [5])
    Fnn_total_fscore, Fnn_avg_fscore, Fnn_total_cEntropy, Fnn_avg_cEntropy = validate.validate_clustering(groundT, [Fivenn], [5])
   
    for i in range(15):
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(images[i]) 
        ax[1].set_title("Kmeans");
        ax[1].imshow(kmeans_results[0][i]);
        ax[2].set_title('Normalized cut');
        ax[2].imshow(Fivenn[i]);
        ax[3].set_title('Groundtruh');
        ax[3].imshow(gt[i][1]); 
        plt.show()   
        print("Average f-score:")
        print("Normalized cut:",Fnn_avg_fscore[0][i])
        print("Kmeans:",kmeans_avg_fscore[0][i])
        if (Fnn_avg_fscore[0][i]<kmeans_avg_fscore[0][i]):
            print("kmeans is better")
        else: print("Normalized cut is better")
        print("\n\n")
        

  
   
   