"""
script for loading image names, keypoints, and descriptors
    from folder.

usage: python load_features.py <folder> 
"""

import sys
import json
import numpy as np

if __name__ == "__main__":

    folder = sys.argv[1]

    if not folder.endswith('/'):
        folder += '/'

    print "=== reading image names ==="
	
    with open(folder+'image_names.json','r') as r:
        image_names = json.load(r)

    print "=== reading image keypoints ==="

    with open(folder+'keypoints.json','r') as r:
        image_keypoints = json.load(r)

    print "=== reading image descriptors ==="

    with open(folder+'descriptors.json','r') as r:
        image_descriptors = json.load(r)

    print image_descriptors[1]

    print "~~~~ finished reading ~~~"
