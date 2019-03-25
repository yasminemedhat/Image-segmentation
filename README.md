# Image-segmentation
Image segmentation using Kmeans.
 Image segmentation means that we can group
similar pixels together and give these grouped pixels the same label. The grouping
problem is a clustering problem. We want to study the use of K-means on the Berkeley
Segmentation Benchmark.

The data is available at the following link.
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bs
ds500.tgz

Every image pixel is a feature vector of 3-dimension {R,G,B}. We will use this feature
representation to do the segmentation.

a. We will change the K of the K-means algorithm between {3,5,7,9,11} clusters.
You will produce different segmentations and save them as colored images. Every
color represents a certain group (cluster) of pixels.
b. We will evaluate the result segmentation using F-measure, Conditional Entropy.
for image I with M available ground-truth segmentations. For a clustering of
K-clusters you will report your measures M times and the average of the M trials
as well. Report average per dataset as well.

In the previous parts, we used the color features RGB. We didn’t encode
the layout of the pixels. We want to modify that for K-means clustering to
encode the spatial layout of the pixels
i. Modify the feature vector to include spatial
layout.
ii. Contrast the results you obtained by k means using RGB only to the results obtained
by considering the spatial layout.
