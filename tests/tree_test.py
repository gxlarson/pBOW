import sys
import unittest
import numpy as np

import sys 
sys.path.append('..')

from tree import (Cluster, constructTree, findChild, Node, Tree)

DATA = np.array([[1,0],[1.1,0],[0.9,0],\
                 [0,0],[0,0.1],[0.1,0],\
                 [0,1],[0,0.9],[0,1.1]])

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

    #def test_construct_tree(self):
    #    """
    #    test tree construction function
    #    """
    #    children = 3
    #    levels = 1
    #    data = DATA
    #    tree_array = constructTree(children, levels, data)
    #    self.assertEqual(len(tree_array), 3)

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
'''
class TestTree(unittest.TestCase):
    def setUp(self):
        self.children = 3
        self.levels = 1
        self.db_desc = [DATA[0:3,:], DATA[3:6,:], DATA[6:9,:]]
        self.db_names = ['im1', 'im2', 'im3']
        self.tree_array = constructTree(self.children,
                                        self.levels,
                                        self.db_desc)
        self.t = Tree(children, levels, tree_array)

    def test_tree_propogate(self):
        """
        tree propogate method test
        """
        pt = self.db_desc[0][0,:]
        index = self.t.propogate(pt)
        in_bound = index < 3 and index >= 0
        assert in_bound is True

    def test_tree_build_tree(self):
        """
        tree object build_tree method tests
        """
        self.t.build_tree(self.db_names, self.db_desc)
'''
if __name__ == '__main__':
    unittest.main()
