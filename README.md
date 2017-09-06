# pBOW

pBOW is a python implementation of an image retrieval pipeline that uses a hierarchical k-means clustering quantization scheme. pBOW uses OpenCV for some of the supporting computer vision algorithms, such as SIFT feature detection and description, and geometric verification using RANSAC.

The pipeline is based on the papers [Scalable Recognition with a Vocabulary Tree (2006)](http://www.cs.ubc.ca/~lowe/525/papers/nisterCVPR06.pdf) and [Mobile Visual Search (2011)](http://reznik.org/papers/SPM11_mobile_visual_search.pdf).

# Image Rretrieval Pipeline

## Hierarchical k-Means Clustering

A hierarchical k-means clustering tree is constructed from the feature descriptors of all "database" images. Given parameters C (children) and L (levels), the feature descriptor space is clustered using k-means into C clusters, forming the first level of the tree. Then, for each cluster, we perform k-means on the feature descriptors contributing to that cluster, and the new cluster centroids are the children of the parent cluster centroid. This process continues until we reach L levels of the tree. The leaf nodes of the tree are called "visual words".

At query time, we propogate each query image descriptor down the tree to find the closest matching leaf node. We keep track of the frequency of each "visual word" in the query image (more precisely, we use tf-idf weighting), and use these frequencies to compute a cosine similarity between the query image and database images. Based on cosine similarity, we return the top m database images.

## Geometric Verification

Given the query image, we re-rank the top m database images based on the inlier count when using RANSAC with SIFT feature keypoints. 

# Sample Images

A sample set of database and query images is located in the `images/bottles` directory. Database images can be found in `images/bottles/database/` directory. These images range from `001` to `100`. Each image is a unique stock photo of a beer bottle label.

Corresponding query images can be found in `images/bottles/query/`. This directory contains 4 sub-directories, called `batch_1` through `batch_4`. Each bottle appears in each batch only once, so there are 400 query images in `images/bottles/query/`. This is the same scheme used by the [Standford Mobile Visual Search dataset (2011)](http://web.cs.wpi.edu/~claypool/mmsys-dataset/2011/stanford/) (i.e. 4 query images per database image).

SIFT features for both database and query images have been pre-computed and are located in the `data/bottles/` directory. This directory contains files `*descriptors.json`, `*image_names.json`, and `*keypoints.json`. If a file is appended with `q*_`, then it is for a query image set. The `*descriptors.json` files contain SIFT descriptors, and the  `*keypoints.json` files contain the corresponding SIFT feature locations (used by geometric verification).

All query images from the `bottles` dataset were collected by the authors of this repository.

# Usage

Use `playground.py` as a configuration file. Here you can set tree parameters `L` and `C`.

Execute the program by entering a python shell and running 

```execfile('playground.py')```.

This will execute `playground.py` and save all variables in the interactive shell, including an object called `matcher`, which is an interface to the hierarchical k-means tree. To query the `matcher`, do

```matcher.query(4)```

which queries using `images/bottles/query/batch_4/004.jpg`. Information returned by this invokation includes query times as well as a ranked list of database images. The ranked list contains tuples of form, e.g., `(u'004.jpg', 62, 15.43419660829007)`, which means database image `004.jpg` had 62 inliers after geometric verification, and had an un-normalized cosine similarity of `15.43` with the query image.

# Requirements

- OpenCV 2.4

- Numpy 1.11

- Python 2.7

# Authors

Stefan Larson and Joshua Kaufman
