"""
feat_extract_and_write.py
author - Stefan Larson
python 2.7
opencv 2.4.12

example usage:
python <path to source directory> SIFT <path to output directory> 
"""

import os
import sys
import json
import cv2


if __name__ == "__main__":
    image_folder = sys.argv[1] # location (directory) of image files 
    feat_detect =  sys.argv[2] # only support SIFT extract & detect for now...
    out_folder 	 = sys.argv[3] # location (directory) where yaml files will go

    if not image_folder.endswith('/'):
        image_folder += '/'

    image_names = [] # list of image file names
    image_keypoints = [] # list of lists of keypoints associated with each image
    image_descriptors = [] # list of lists of descriptors

    features = cv2.SIFT() # opencv SIFT feature extractor

    for im in os.listdir( image_folder ):
        if im.startswith('.'):
            continue

        pic = cv2.imread(image_folder + im, 0)

        # extract features from pic
        # keypoints: x,y locations
        # descriptors: feature vectors
        keypoints, descriptors = features.detectAndCompute(pic, None)
        
        # convert from np.float32 to in
        descriptors = descriptors.astype(int)

        descriptors = descriptors.tolist()
        
        # just get the x,y locations:
        keypoints = [kp.pt for kp in keypoints]
	
        print 'Extracting and writing features from: ' + im

        n_features = len(keypoints)

        print '- - - extracted ' + str(n_features) + ' features'

        image_names.append(im)
        image_keypoints.append(keypoints)
        image_descriptors.append(descriptors)	

    print '=== Writing features to files ==='

    if not out_folder.endswith('/'):
        out_folder += '/'

    print '=== Writing features to json files ==='

    with open(out_folder+'image_names.json','w') as w:
        json.dump(image_names, w)
    with open(out_folder+'keypoints.json','w') as w:
        json.dump(image_keypoints, w)
    with open(out_folder+'descriptors.json','w') as w:
        json.dump(image_descriptors, w)	
