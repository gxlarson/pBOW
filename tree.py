"""
objects and functions for building hierarchical k-means clustering tree.
"""

import sys
import math
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import deque
#from kmeans import *


class Cluster(object):
    """
    Used within the hierarchical k-means tree.
    Cluster object is a struct containing:
        i: node id (index within treeArray)
        l: level within the tree
        data: collection of points within cluster
    """
    def __init__(self, i, l, data):
        self.i = i
        self.l = l
        self.data = data


class Node(object):
    """
    node class. a node is an element in treeArray
        cen: centroid
        index: position in treeArray
        inverted_index: becomes a dict when used as an inverted index
        with imageID and word frequency (tf) pairs
    """
    def __init__(self):
        self.cen = None
        self.index = None
        self.inverted_index = None # becomes a dict later


class Tree(object):
    """
    Hierarchical k-means tree.
        C: num children per node
        L: num levels in tree (root node not included)
        treeArray: 1-d array of node objects
    """

    def __init__(self, C, L, treeArray):
        self.treeArray = treeArray
        self.L = L # levels in tree
        self.C = C # branching factor: children per internal node
        self.N = 0 # num images contributing to tree
        self.imageIDs = []
        self.dbLengths = {}

    def build_tree(self, db_names, db_descriptors):
        """
        db_names: list of image names
        db_descriptors: list of db image feature descriptors
        """
        f_i = 0
        for im_id in db_names:
            self.fill_tree(im_id,db_descriptors[f_i])
            f_i += 1
        self.set_lengths()

    def propagate(self, pt):
        """
        propogate a feature descriptor (pt) down the tree until
        reaching leaf node. Path down tree is dictated by euclidean
        distance between pt and cluster centroids (tree nodes).
        Array index of leaf node is returned (i).
        """
        i = 0 # initialize position to top node
        l = 0 # initialize position to top level
        closeChild = 0 
        while l != self.L:
            curDist = np.inf
            minDist = np.inf
            for x in range(0,self.C):
                childPos = findChild(self.C,i,x)
                testPT = self.treeArray[childPos].cen
                if testPT == None:
                    continue
                # euclidean distance between child x and pt
                curDist = np.linalg.norm(testPT - pt)
                if curDist < minDist:
                    minDist = curDist 
                    closeChild = childPos
            i = closeChild
            l += 1
        return i

    def fill_tree(self, imageID, features):
        """
        updates inverted indexes with imageID and corresponding features
        """
        for feat in features:
            # quantize feat to leaf node
            leaf_node = self.propagate(feat) 
            # add to inverted index
            if imageID not in self.treeArray[leaf_node].inverted_index:
                self.treeArray[leaf_node].inverted_index[imageID] = 1
            else:
                self.treeArray[leaf_node].inverted_index[imageID] += 1
        self.N += 1 # increase num images contributing to tree
        self.imageIDs.append(imageID)

    def set_lengths(self):
        """
        find database image vector lengths (used in score normalization)
        """
        # process db vector lengths:
        num_nodes = len(self.treeArray)
        num_leafs = self.C ** self.L
        for imageID in self.imageIDs:
            cum_sum = float(0)
            # iterate over only leaf nodes:
            for lf in range(num_nodes-1, num_nodes-num_leafs-1, -1):
                if self.treeArray[lf].inverted_index == None:
                    continue
                if imageID in self.treeArray[lf].inverted_index:
                    # tf is frequency of lf in imageID
                    tf = self.treeArray[lf].inverted_index[imageID]
                    # df is num images containing lf visual word
                    df = len(self.treeArray[lf].inverted_index)
                    idf = math.log( float(self.N) / float(df) )
                    cum_sum += math.pow( tf*idf , 2)
            self.dbLengths[imageID] = math.sqrt( cum_sum )

    def process_query(self, features, n):
        """
        features: list of features in query image
        n: return top n scores
        """
        scores = {} # dict of imageID to score
        for feat in features:
            leaf_node = self.propagate(feat)
            idx = self.treeArray[leaf_node].inverted_index.items()
            for (ID,count) in idx:
                df = len(idx) # document frequency in inverted index
                idf = math.log( float(self.N) / float(df) )
                idf_sq = idf * idf
                tf  = count
                score = float(tf * idf_sq)
                if ID not in scores:
                    scores[ID] = score
                else:
                    scores[ID] += score
        # normalize scores by scaling by norm of db vectors
        scores = scores.items()
        final_scores = [] # TODO: change this to a heap so it can sort by score
        for i in range(len(scores)):
            (ID,score) = scores[i]
            nmz_score = float(score) / float(self.dbLengths[ID])
            # TODO: change this to push onto heap so we can auto sort
            final_scores.append((ID, nmz_score))
        # sort final scores and return
        final_scores.sort(key=lambda pair : pair[1], reverse=True)
        return final_scores[0:n]


def findChild(C,i,x):
    """
    returns the x^th child of node i. x ranges from [0,C)
    """
    return (C*(i+1)-(C-2)+x-1)


def constructTree(C, L, data):
    """
    C: # children per node
    L: # levels in tree (root is not considered a level)
    data: numpy.ndarray of size m x n 
        ... where n is length of feature descriptor
        m is number of features in the database image set
    """
    print "building tree: C = " + str(C) + ", L = " + str(L)

    NUM_NODES = (C**(L+1)-1)/(C-1)

    # initialize tree array with empty nodes
    treeArray = [Node() for i in range(NUM_NODES)]
    NUM_LEAFS = 0

    # KMEANS PARAM INPUTS
    cv2_iter = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (cv2_iter, 10, 1.0)

    queue = deque()
    queue.appendleft( Cluster(0,0,data) )

    while len(queue):
        clust = queue.pop()
        # KMEANS FUNCTION CALL
        # let N be then number of points we seek to cluster using KMeans
        # let d be the dimensionality of each data point
        # compactness is not used by us
        # label: Nx1 numpy array with labels \in [0,C)
        # center: Cxd numpy array with rows as cluster centroids

        if C <= len(clust.data):
            # on mac (opencv 2.4.5):
            compactness, label, center = cv2.kmeans(clust.data,
                                                   C,
                                                   criteria,
                                                   10,
                                                   0)
            # on CAEN (opencv 3.1.0):
            #compactness, label, center=cv2.kmeans(clust.data,
            #                                      C,
            #                                      None,
            #                                      criteria,
            #                                      10,
            #                                      cv2.KMEANS_RANDOM_CENTERS)
            # custom kmeans implementation:
            #label, center = kmeans(cv2.TERM_CRITERIA_EPS,
            #                       cv2.TERM_CRITERIA_MAX_ITER,
            #                       clust.data,
            #                       C)

            if clust.l+1 != L:
                # print "NOT LEAF"
                for x in range(0,C):
                    childPos = findChild(C,clust.i,x)
                    # print childPos
                    queue.appendleft(Cluster(childPos,
                                             clust.l+1,
                                             clust.data[label.ravel()==x]))
                    treeArray[childPos].cen = center[x,:]
            else:
                # print "LEAF"
                for x in range(0,C):
                    childPos = findChild(C,clust.i,x)
                    # print childPos
                    #treeArray[childPos].index = clust.data[label.ravel()==x]
                    treeArray[childPos].inverted_index = {}
                    treeArray[childPos].cen = center[x,:]
                    if clust.data.size == 0:
                        print "ZERO CLUSTER ========="
                    NUM_LEAFS += 1
        else:
            # pass down data to first (0th) child; 
            # pass down Nones to other children 
            # (aka do nothing since they were initialized with None)
            x = 0
            childPos = findChild(C,clust.i,x)
            treeArray[childPos].cen = np.zeros(len(clust.data[0,:]),
                                               dtype='float32')
            if clust.l+1 != L:
                queue.appendleft(Cluster(childPos,
                                         clust.l+1,
                                         clust.data))
            else:
                treeArray[childPos].inverted_index = {}

    print "num leafs: " + str(NUM_LEAFS)
    return treeArray
