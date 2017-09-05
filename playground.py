"""
example usage script
"""

from matcher import *

# parameters for building hierarchical k-means tree
L = 4 # num levels in tree (depth)
C = 10 # branching factor (num children per each node)

dataset = "bottles" # one of {"bottles","books","paintings"}


# load database information
print "--- loading database images ---"
with open('data/' + dataset + '/image_names.json','r') as r:
    # list of database image names
    image_names = json.load(r)
with open('data/' + dataset + '/descriptors.json','r') as r:
    # list of database image descriptors 
    image_descriptors = json.load(r)
with open('data/' + dataset + '/keypoints.json','r') as r:
    image_keypoints = json.load(r)


# load query information
print "---- loading query images ----"
q_descriptors = []
with open('data/' + dataset + '/q4_descriptors.json','r') as r:
    q_descriptors += json.load(r)
q_ids = []
with open('data/' + dataset + '/q4_image_names.json','r') as r:
    q_ids += json.load(r)
q_kps = []
with open('data/' + dataset + '/q4_keypoints.json','r') as r:
    q_kps += json.load(r)



FEATS = []
# make FEATS list into a list of float32; each element in list is a feature descriptor
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
