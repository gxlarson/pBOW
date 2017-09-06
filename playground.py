"""
example usage script
"""

import sys
from matcher import *
from utilities.utils import load_data

# parameters for building hierarchical k-means tree
L = 4 # num levels in tree (depth)
C = 10 # branching factor (num children per each node)

dataset = "bottles" # one of {"bottles","books","paintings"}

query_set = 4 # one of {1,2,3,4}

# ---------------------------------------------------------------------

# load database information
print "--- loading database images ---"
(image_names, image_descriptors, image_keypoints) = \
    load_data('database', 'bottles')

# load query information
print "---- loading query images ----"
(q_ids, q_descriptors, q_kps) = load_data('query',
                                          'bottles',
                                          query_set)

FEATS = []
# make FEATS list into a list of float32; 
# each element in list is a feature descriptor
for feats in image_descriptors:
    FEATS += [np.array(fv,dtype='float32') for fv in feats]

# make FEATS a numpy array (each row is a feature descriptor)
FEATS = np.vstack(FEATS)

treeArray = constructTree(C, L, np.vstack(FEATS))
tree = Tree(C, L, treeArray)
tree.build_tree(image_names, image_descriptors)
tree.set_lengths()

matcher = Matcher(image_descriptors, image_keypoints, image_names)
matcher.update_tree(tree)
matcher.add_queries(q_descriptors, q_kps)
