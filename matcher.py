"""
funcitons for querying a database with descriptors from a query
image.
"""

import time
from tree import *

MIN_MATCH_COUNT = 10
N_DECIMAL = 4


def gv_pair(q_kp, q_des, db_kp, db_des):
    """
    Perform geometric verification on a query-database
        image pair.
    q_kp: query image feature keypoints (locations)
    q_des: query image feature descriptors
    db_kp: database image feature keypoints (locations)
    db_des: database image feature descriptors
    M: homography matrix
    n_inliers: number of inliers for best consensus model using RANSAC
    ransac_matches: features deemed as matches after gv
    """
    M = None
    n_inliers = 0
    ransac_matches = []

    q_des = np.asarray(q_des,np.float32)
    db_des= np.asarray(db_des,np.float32)

    matcher = cv2.BFMatcher() # brute-force matcher
    matches = matcher.knnMatch(q_des, db_des, 2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good_matches.append([m])
    # find & compute homography if there are enough good_matches
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = [q_kp[m[0].queryIdx] for m in good_matches]
        src_pts = np.float32(src_pts).reshape(-1,1,2)
        dst_pts = [db_kp[m[0].trainIdx] for m in good_matches]
        dst_pts = np.float32(dst_pts).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return M, n_inliers, ransac_matches
        matchesMask = mask.ravel().tolist()
        n_inliers = len( [1 for m in matchesMask if m==1] )
        ransac_matches = [good_matches[i] for i in range(len(matchesMask)) \
                          if matchesMask[i]==1]
    return M, n_inliers, ransac_matches


class Matcher(object):
    """
    module for finding match given query and database descriptors.
    """
    def __init__(self, db_des, db_kp, db_names):
        self.tree = None
        self.db_descriptors = db_des
        self.db_keypoints = db_kp
        self.db_names = db_names
        self.name2id = {}
        for i in range(len(self.db_names)):
            self.name2id[self.db_names[i]] = i

    def update_tree(self, in_tree):
        """
        set the tree
        """
        self.tree = in_tree

    def query_features(self, q_des, q_kp, m=5, r=5):
        """
        q_des: features of query image
        q_kp: query feature keypoints
        m: do geometric reranking on top m from BOW list
        r: return ranked list of r results to user
        """
        bow_list, bow_time = self.bow_score(q_des, q_kp, m)
        gv_list, gv_time = self.geometric_reranking(bow_list, q_des, q_kp)
        print "BOW: " + str(bow_time)
        print "GV : " + str(gv_time)
        return gv_list[0:r], bow_time, gv_time

    def bow_score(self, q_des, q_kp, m):
        """
        find the top-m sim scores given query kp & des.
        also computes query time.
        """
        start_time = time.time()
        bow_list = self.tree.process_query(q_des,m)
        bow_time = round(time.time() - start_time, N_DECIMAL)
        return bow_list, bow_time

    def geometric_reranking(self, bow_list, q_des, q_kp):
        """
        perform geometric re-ranking using ransac
        """
        gv_list = []
        start_time = time.time()
        for (name,sim_score) in bow_list:
            db_id = self.name2id[name]
            db_des = self.db_descriptors[db_id]
            db_kp = self.db_keypoints[db_id]
            H, n_inliers, matches = gv_pair(q_kp, q_des, db_kp, db_des)
            gv_score = n_inliers
            gv_list.append( (name,gv_score,sim_score) )
        gv_list.sort(key=lambda gv: gv[1], reverse=True)
        gv_time = round(time.time()-start_time,4)
        return gv_list, gv_time

    def add_queries(self, q_descriptors, q_kps):
        """
        update q_kps and q_descriptors
        """
        self.nq = len(q_kps)
        self.q_descriptors = q_descriptors
        self.q_kps = q_kps

    def query(self, q_id):
        """
        query using the id number of the query image
        """
        assert isinstance(q_id, int)
        if q_id > self.nq or q_id < 0:
            print "query id out of range"
            return
        q_kp = self.q_kps[q_id-1]
        q_des= self.q_descriptors[q_id-1]
        return self.query_features(q_des, q_kp, 5, 5)
