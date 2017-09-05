import json
    
import yaml
import numpy as np
import sys

if __name__ == "__main__":
	
	folder = sys.argv[1]
	
	if not folder.endswith('/'):
		folder += '/'
	
	print "=== reading image names ==="
	#with open(folder+'image_names.yaml','r') as r:
	#	image_names = yaml.load(r)
	
	#with open(folder+'image_names.pickle','rb') as r:
	#	image_names = pickle.load(r)
		
	with open(folder+'image_names.json','r') as r:
		image_names = json.load(r)
	
	print "=== reading image keypoints ==="
	#with open(folder+'keypoints.yaml','r') as r:
	#	image_keypoints = yaml.load(r)
	
	#with open(folder+'keypoints.pickle','rb') as r:
	#	image_keypoints = pickle.load(r)
		
	with open(folder+'keypoints.json','r') as r:
		image_keypoints = json.load(r)
		
	print "=== reading image descriptors ==="
	#with open(folder+'descriptors.yaml','r') as r:
	#	image_descriptors = yaml.load(r)
	
	#with open(folder+'descriptors.pickle','rb') as r:
	#	image_descriptors = pickle.load(r)
		
	with open(folder+'descriptors.json','r') as r:
		image_descriptors = json.load(r)
	
	print image_descriptors[1]
	
	print "~~~~ finished reading ~~~"