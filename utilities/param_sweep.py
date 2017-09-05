import cv2
import json
import numpy as np
from build_tree import *

NUM_QUERIES = 400

# load database information
print "--- loading database images ---"
with open('big_data/image_names.json','r') as r:
	# list of database image names
	image_names = json.load(r)
with open('big_data/descriptors.json','r') as r:
	# list of database image descriptors 
	image_descriptors = json.load(r)

# load query information
# with open('big_data/q_names.json','r') as r:
# 	q_names = json.load(r)
q_descriptors = []
with open('big_data/q1_descriptors.json','r') as r:
	q_descriptors += json.load(r)
with open('big_data/q2_descriptors.json','r') as r:
	q_descriptors += json.load(r)
with open('big_data/q3_descriptors.json','r') as r:
	q_descriptors += json.load(r)
with open('big_data/q4_descriptors.json','r') as r:
	q_descriptors += json.load(r)
q_ids = []
with open('big_data/q1_image_names.json','r') as r:
	q_ids += json.load(r)
with open('big_data/q2_image_names.json','r') as r:
	q_ids += json.load(r)
with open('big_data/q3_image_names.json','r') as r:
	q_ids += json.load(r)
with open('big_data/q4_image_names.json','r') as r:
	q_ids += json.load(r)


FEATS = []
# make FEATS list into a list of float32; each element in list is a feature descriptor
for feats in image_descriptors:
	FEATS += [np.array(fv,dtype='float32') for fv in feats]

# make FEATS a numpy array (each row is a feature descriptor)
FEATS = np.vstack(FEATS)
"""
# loop thru parameters C and L and write results to JSON file
# with open('results.json','w') as w:
results_write = open('results.txt','w')
	
for L in [4,5,6,7]:

	for C in [6,7,8,9,10]:
	
		print "===== running test for L = " + str(L) + "; C = " + str(C) + " ====="
		
# 		sweep_dict = {'L':L, 'C':C}
	
		# make tree array
		treeArray = constructTree(C,L,np.vstack(FEATS))
	
		# init tree
		tree = Tree(C,L,treeArray)
	
		# fill tree leaf nodes with database image descriptors (fill inverted indexes)
		f_i = 0
		for id in image_names:
			tree.fill_tree(id,image_descriptors[f_i])
			f_i += 1
	
		# set database lengths (used in score normalization at query time)
		tree.set_lengths()
	
		# process queries
		ranks = []
		for qi in range(NUM_QUERIES):
			f_id = q_ids[qi]
			id = f_id.split('.')[0]
			print "processing query ID: " + id
			q = q_descriptors[qi]
			ranked_list = tree.process_query(q,100)
			n_ranked = len(ranked_list)
			for rk in range(n_ranked):
				dbID = ranked_list[rk][0]
				dbID = int( dbID.split('.')[0] )
				if dbID == int(id):
					print "rank = " + str(rk + 1)
					ranks.append(rk + 1)
# 			print ranked_list[0:3]
		
		hits = sum( [1 for rank in ranks if rank <= 10] )
		acc10 = float(hits) / float(NUM_QUERIES)
		hits = sum([1 for rank in ranks if rank == 1])
		acc1 = float(hits) / float(NUM_QUERIES)
		hits = sum([1 for rank in ranks if rank <= 5])
		acc5 = float(hits) / float(NUM_QUERIES)
		
# 		sweep_dict['acc'] = acc
				
		results = 'L: ' + str(L) + ' C: ' + str(C) + ' acc1: ' + str(acc1) + ' acc5: ' + str(acc5)   + ' acc10: ' + str(acc10) 
		print results
		
		results_write.write(results)
		results_write.write('\n')
		
# 		json.dump(sweep_dict,w)
results_write.close()
	"""					
