import sys
import unittest
import numpy as np
import sys 
from tree import (Cluster, constructTree, findChild, Node, Tree)
from matcher import Matcher
from utilities.utils import load_data


class TestSimpleTree(unittest.TestCase):
    """
    tests for objects and methods in tree.py
    """

    def test_cluster_init(self):
        """
        cluster object init tests
        """
        i = 7
        l = 3
        data = [1,2,3]
        c = Cluster(i, l, data)
        self.assertEqual(i, c.i)
        self.assertEqual(l, c.l)
        assert isinstance(c.data, list)

    def test_node_init(self):
        """
        node object init tests
        """
        n = Node()
        self.assertEqual(n.cen, None)
        self.assertEqual(n.index, None)
        self.assertEqual(n.inverted_index, None)

    def test_tree_init(self):
        """
        tree object init tests
        """
        children = 3
        levels = 4
        array = [Node(), Node(), Node()]
        t = Tree(children, levels, array)
        self.assertEqual(t.C, children)
        self.assertEqual(t.L, levels)
        assert isinstance(t.treeArray, list)
        assert isinstance(t.dbLengths, dict)
        assert isinstance(t.imageIDs, list)

    def test_find_child(self):
        """
        test find_child function
        """
        children = 3
        node = 0
        index = findChild(children, node, 1)
        self.assertEqual(index, 2)
        children = 5
        node = 10
        index = findChild(children, node, 1)
        self.assertEqual(index, 52)

    def test_tree(self):
        """
        full test of tree construction and query
        """
        L = 4 # num levels in tree (depth)
        C = 10 # branching factor (num children per each node)
        dataset = "bottles" # one of {"bottles","books","paintings"}
        (image_names, image_descriptors, image_keypoints) = \
            load_data('database', 'bottles')
        (q_ids, q_descriptors, q_kps) = load_data('query', 'bottles', 4)
        features = []
        for feats in image_descriptors:
            features += [np.array(fv,dtype='float32') for fv in feats]
        features = np.vstack(features)
        treeArray = constructTree(C, L, np.vstack(features))
        t = Tree(C, L, treeArray)
        t.build_tree(image_names, image_descriptors)
        t.set_lengths()
        matcher = Matcher(image_descriptors, image_keypoints, image_names)
        matcher.update_tree(t)
        matcher.add_queries(q_descriptors, q_kps)
        result = matcher.query(4)
        print result
        result_im = str(result[0][0][0])
        self.assertEqual(result_im, '004.jpg')

if __name__ == '__main__':
    unittest.main()
