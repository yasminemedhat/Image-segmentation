import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.io import loadmat
from sklearn.cluster import KMeans
import imageio
from validation import validate_clustering

#read images and return two lists; images and their names, each image is 381*481
def read_images(path):
    #list to hold images
    images = []
    #take onlt jpg files
    files = [file[:-4] for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file.endswith('jpg') ]
    files.sort(key=int)
    for file in files:
        file = file + '.jpg'
        images.append(img.imread(os.path.join(path, file))) 
    return files, images

#read the ground truth images for each image
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
    fig, ax = plt.subplots(1, n, figsize=(15,15))
    ax[0].imshow(images[index])
    for i in range(len(gt[index])):
        ax[i+1].imshow(gt[index][i])
    plt.show()
        
#segmentation using kmeans and return the list of images segmented to different clusters
def Kmeans(data, n_clusters):
    clusters = []
    i = 0
    for image in data:
        #unroll image and represent it as one row of pixels eac pixel has 3 dimensions RGB
        img_unrolled = image.reshape(image.shape[0]*image.shape[1],image.shape[2])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_unrolled)
        #append the labels after reshaping to the shape of the real image 321*481
        #kmeans.labels_.reshape(image.shape[:-1]))
        
        clusters.append(kmeans.labels_.reshape(image.shape[:-1]))
        i = i + 1
        if i % 25 == 0:
            print(i*2,"%" ,"of images clustered")
    return clusters 

#compute kmeans for several different number of clusters
def diff_kmeans(data, n_clusters_list):
    results = []
    for k in n_clusters_list:
        print("computing clusters for k = ", k)
        results.append(Kmeans(data, k))
    return results

#read saved kmeans clustering results directly
def read_kmeans_results(path):
    files = [f[:-4] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('png')]
    files.sort(key=int)
    results = []
    for file in files:
        file = file + '.png'
        total_path = os.path.join(path,file)
        img = imageio.imread(total_path)
        results.append(img)
    return results
 
#display a selected image with the segmented image after performing kmeans clustering wit each k in k_list
def display_img_with_segmented_img(images, filenames, clustered_imgs, index, k_list):
    #get number of images to be displayed; real image plus segmented image for each k in kmeans
    n = len(k_list)+1
    fig, ax = plt.subplots(1, n, figsize=(20,5))
    fig.suptitle('image: ' + filenames[index])
    ax[0].imshow(images[index]) 
    for i in range(len(k_list)):
        ax[i+1].set_title('K = ' + str(k_list[i]));
        ax[i+1].imshow(clustered_imgs[i][index]);
    plt.show()   


def main():  
    path1 = r"E:\Documents\Term 8\Pattern Recognition\assignments\proj2\images"   
    path2 = r"E:\Documents\Term 8\Pattern Recognition\assignments\proj2\groundTruth"
    img_names, images = read_images(path1)
    gt = read_groundTruth(path2)
    for i in range(50):
        display_img_with_groundT(images, gt, i)
   
    k_list = [3, 5, 7, 9, 11]
    kmeans_results = diff_kmeans(images, k_list)
    
    for i in range(50):
        display_img_with_segmented_img(images, img_names, kmeans_results, i, k_list)
    #get scores
    total_fscore, avg_fscore, total_cEntropy, avg_cEntropy = validate_clustering(gt, kmeans_results, k_list)
    #plot validation results
    for i in range(len(k_list)):
        plt.figure(figsize=(20,10))
        plt.title('K = ' + str(k_list[i]))
        plt.plot(avg_fscore[i], label = 'Average F-measure')
        plt.plot(avg_cEntropy[i], label = 'Average Conditional Entropy')
        plt.legend()
        plt.show()
        print('With k = ' , k_list[i])
        print('Average F-measure over dataset : ', np.mean(avg_fscore[i]))
        print('Average Conditional entropy over dataset : ', np.mean(avg_cEntropy[i]))