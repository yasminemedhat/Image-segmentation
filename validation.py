# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:45:40 2019

@author: lenovo
"""

from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import math

#return the fmeasure of ground truth and kmeans-clustered image
def f_measure(contigency_matrix):
    fscore = 0
    for i in range(len(contigency_matrix[0])):
        maxi = np.argmax(contigency_matrix[:,i])
        ni = np.sum(contigency_matrix[:,i])
        prec = contigency_matrix[maxi][i] / ni
        recall = contigency_matrix[maxi][i] / np.sum(contigency_matrix[maxi,:])
        fscorei = (2 * prec * recall) / (prec + recall)
        fscore += fscorei
    return fscore/len(contigency_matrix[0])

def conditional_entropy(contigency_matrix, n):
    cEntropy = 0 
    for i in range(len(contigency_matrix[0])):
         entropy = 0
         ni = np.sum(contigency_matrix[:,i])
         for t in range(len(contigency_matrix)):
             probT = contigency_matrix[t][i] / ni
             if (probT != 0.0):
                 entropy -= probT * math.log2(probT)
         entropy *=  ni / n
         cEntropy += entropy
    return cEntropy
    

def validate_clustering(groundTruths, results, k_list):
    #fmeasure and cond. entropy for all k parameters for all images, each image with M ground truths 
    total_fscore = []
    total_cEntropy = []
    #average of M fmeasure and cond. entropy for all k parameters for all images truths
    tot_avg_fscore = []
    tot_avg_cEntropy = []
    for k in range(len(k_list)):
        fscore = []
        cond_entr = []
        avg_fscore = []
        avg_cEnt = []
        i = 0
        #for each segmented image
        for result_img in results[k]:
            f = []
            cent = []
            img = result_img.reshape(-1)
            #get fscore and entropy for each gt available for this image
            for gt in groundTruths[i]:
                gt = gt.reshape(-1)
                #get the contingency matrix of the gt and the segmented image
                contingency_mat= contingency_matrix(gt, img)
                f.append(f_measure(contingency_mat))
                cent.append(conditional_entropy(contingency_mat, len(gt)))
            fscore.append(f)
            cond_entr.append(cent)
            avg_fscore.append(np.mean(f))
            avg_cEnt.append(np.mean(cent))
            i = i+ 1
        total_fscore.append(fscore)
        total_cEntropy.append(cond_entr)
        tot_avg_fscore.append(avg_fscore)
        tot_avg_cEntropy.append(avg_cEnt)
    return total_fscore,tot_avg_fscore, total_cEntropy, tot_avg_cEntropy



