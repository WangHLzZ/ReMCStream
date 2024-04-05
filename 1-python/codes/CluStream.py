from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from MicroCluster import MicroCluster as model
from scipy.spatial import distance
import math
import timeit
import numpy as np
import threading
import time
import sys
# from config import *


class CluStream(BaseEstimator, ClusterMixin):
    # Implementation of CluStream

    def __init__(self, maxMcs, maxUMcs, nb_initial_points, numK, time_window=1000, timestamp=0, clocktime=0, 
                 nb_macro_cluster=5, micro_clusters=[], alpha=2, l=2, h=1000):
        self.start_time = time.time()
        self.nb_initial_points = nb_initial_points # eachClass
        self.time_window = time_window  # Range of the window
        self.timestamp = timestamp
        # self.newtimestamp = 0
        # self.clocktime = clocktime
        # self.micro_clusters = micro_clusters
        self.labeled_micro_clusters = []
        self.unlabeled_micro_clusters = []
        self.nb_macro_cluster = nb_macro_cluster
        # self.alpha = alpha
        # self.l = l
        # self.h = h
        #self.snapshots = []
        self.nb_created_clusters = 0
        self.maxnb_mc = maxMcs
        self.maxUMcs = maxUMcs
        self.avg_radius = 0
        self.numK = numK

    def fit(self, X, labelClu, Y=None):
        X = check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points >= self.nb_initial_points:
            kmeans = KMeans(n_clusters=self.nb_initial_points, random_state=1)
            micro_cluster_labels = kmeans.fit_predict(X, Y)
            X = np.column_stack((micro_cluster_labels, X))  
            initial_clusters = [X[X[:, 0] == l][:, 1:]
                                for l in set(micro_cluster_labels) if l != -1]
            for cluster in initial_clusters:
                self.create_micro_cluster(cluster, labelClu)
        elif nb_initial_points >= 1:
            kmeans = KMeans(n_clusters=nb_initial_points, random_state=1)
            micro_cluster_labels = kmeans.fit_predict(X, Y)
            X = np.column_stack((micro_cluster_labels, X))  
            initial_clusters = [X[X[:, 0] == l][:, 1:]
                                for l in set(micro_cluster_labels) if l != -1]
            for cluster in initial_clusters:
                self.create_micro_cluster(cluster, labelClu)
        self.start_time = time.time()

    def create_micro_cluster(self, cluster, labelClu):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        self.nb_created_clusters += 1
        new_m_cluster = model(identifier=self.nb_created_clusters, nb_points=0, linear_sum=linear_sum,
                              squared_sum=squared_sum, create_timestamp=0, labelMc=labelClu)
        for point in cluster:
            new_m_cluster.insert(point, self.timestamp)
        
        if labelClu != -1:
            self.labeled_micro_clusters.append(new_m_cluster)
        else:
            self.unlabeled_micro_clusters.append(new_m_cluster)
            if len(self.unlabeled_micro_clusters) > self.maxUMcs:
                self.del_oldest_UnlabelMc()

    def distance_to_cluster(self, x, cluster):
        return distance.euclidean(x, cluster.get_center())

    def find_closest_cluster(self, x, micro_clusters):
        min_distance = sys.float_info.max
        for cluster in micro_clusters:
            distance_cluster = self.distance_to_cluster(x, cluster)
            if distance_cluster < min_distance:
                min_distance = distance_cluster
                closest_cluster = cluster
        return closest_cluster, min_distance

    def check_fit_in_cluster(self, x, cluster):
        if cluster.get_weight() == 1:
            # determine radius using next closest micro-cluster
            # radius = sys.float_info.max
            # micro_clusters = self.labeled_micro_clusters.copy()
            # micro_clusters_2 = self.unlabeled_micro_clusters.copy()
            # micro_clusters += micro_clusters_2
            # micro_clusters.remove(cluster)
            # next_cluster, dis = self.find_closest_cluster(x, micro_clusters)
            # dist = distance.euclidean(
            #     next_cluster.get_center(), cluster.get_center())
            # radius = min(dist, radius)
            radius = cluster.get_radius()
        else:
            radius = cluster.get_radius()
        if self.distance_to_cluster(x, cluster) < radius:
            return True
        else:
            return False

    def oldest_updated_cluster(self, micro_clusters):
        threshold = self.timestamp - self.time_window
        min_relevance_stamp = sys.float_info.max
        oldest_cluster = None
        for cluster in micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if (relevance_stamp < threshold) and (relevance_stamp < min_relevance_stamp):
                min_relevance_stamp = relevance_stamp
                oldest_cluster = cluster
        return oldest_cluster

    def merge_closest_labeled_clusters(self):
        min_distance = sys.float_info.max
        for i, cluster in enumerate(self.labeled_micro_clusters):
            center = cluster.get_center()
            for next_cluster in self.labeled_micro_clusters[i+1:]:
                dist = distance.euclidean(center, next_cluster.get_center())
                if dist < min_distance:
                    min_distance = dist
                    cluster_1 = cluster
                    cluster_2 = next_cluster
        assert (cluster_1 != cluster_2)
        if cluster_1.labelMc == cluster_2.labelMc:
            cluster_1.merge(cluster_2)
            self.labeled_micro_clusters.remove(cluster_2)
            return True
        else:
            return False
    
    def del_oldest_UnlabelMc(self):
        min_relevance_stamp = sys.float_info.max
        oldest_cluster = None
        for cluster in self.unlabeled_micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if relevance_stamp < min_relevance_stamp:
                min_relevance_stamp = relevance_stamp
                oldest_cluster = cluster
        self.unlabeled_micro_clusters.remove(oldest_cluster)

    def del_oldest_LabelMc(self):
        min_relevance_stamp = sys.float_info.max
        oldest_cluster = None
        for cluster in self.labeled_micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if relevance_stamp < min_relevance_stamp:
                min_relevance_stamp = relevance_stamp
                oldest_cluster = cluster
        self.labeled_micro_clusters.remove(oldest_cluster)

    def del_oldest_Mc(self):
        min_relevance_stamp1 = sys.float_info.max
        min_relevance_stamp2 = sys.float_info.max
        oldest_cluster1 = None
        oldest_cluster2 = None
        oldest_cluster = None
        for cluster in self.labeled_micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if relevance_stamp < min_relevance_stamp1:
                min_relevance_stamp1 = relevance_stamp
                oldest_cluster1 = cluster
        for cluster in self.unlabeled_micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if relevance_stamp < min_relevance_stamp2:
                min_relevance_stamp2 = relevance_stamp
                oldest_cluster2 = cluster
        if (min_relevance_stamp1 < min_relevance_stamp2) and (len(self.labeled_micro_clusters)>self.numK):
            oldest_cluster = oldest_cluster1
            self.labeled_micro_clusters.remove(oldest_cluster)
        else:
            oldest_cluster = oldest_cluster2
            self.unlabeled_micro_clusters.remove(oldest_cluster)
        
    def creat(self, x, label, predict_idx=None):
        X = x
        linear_sum = np.zeros(len(X))
        squared_sum = np.zeros(len(X))
        if ((len(self.labeled_micro_clusters) + len(self.unlabeled_micro_clusters)) >= self.maxnb_mc):
            self.del_oldest_Mc()

        self.nb_created_clusters += 1
        new_m_cluster = model(identifier=self.nb_created_clusters, nb_points=0, linear_sum=linear_sum,
                              squared_sum=squared_sum, create_timestamp=self.timestamp, labelMc=label)
        new_m_cluster.insert(X, self.timestamp)
        
        new_m_cluster.radius = self.calculate_avg_radius()
        if label != -1:
            new_m_cluster.labelType = 1
            self.labeled_micro_clusters.append(new_m_cluster)
        elif predict_idx!=None:
            new_m_cluster.labelMc = predict_idx
            new_m_cluster.labelType = 0
            self.labeled_micro_clusters.append(new_m_cluster)
        else:
            if len(self.unlabeled_micro_clusters) >= self.maxUMcs:
                self.del_oldest_UnlabelMc()
                assert(1==2)
            self.unlabeled_micro_clusters.append(new_m_cluster)

    def calculate_avg_radius(self):
        micro_clusters = self.labeled_micro_clusters.copy()
        micro_clusters_2 = self.unlabeled_micro_clusters.copy()
        micro_clusters += micro_clusters_2
        centers = [mc.get_radius() for mc in micro_clusters if mc.nb_points > 1]
        if len(centers) > 1:
            self.avg_radius = np.average(np.array([mc.get_radius() for mc in micro_clusters if mc.nb_points > 1]))
            for mc in self.labeled_micro_clusters:
                if mc.nb_points <= 1:
                    mc.radius = self.avg_radius
            for mc in self.unlabeled_micro_clusters:
                if mc.nb_points <= 1:
                    mc.radius = self.avg_radius
        else:
            self.avg_radius = np.average(np.array([mc.get_radius() for mc in micro_clusters]))
        return self.avg_radius
    
#     def partial_fit(self, x, y=None):
#         self.timestamp += 1
#         X = x
# #         x = x[0]
#         closest_cluster = self.find_closest_cluster(x, self.micro_clusters)
#         check = self.check_fit_in_cluster(x, closest_cluster)
#         if check:
#             #             print('insert')
#             closest_cluster.insert(x, self.timestamp)
#         else:
#             old_up_clust = self.oldest_updated_cluster()
#             if old_up_clust is not None:
#                 #                 print('remove')
#                 self.micro_clusters.remove(old_up_clust)
#             else:
#                 #                 print('merge')
#                 self.merge_closest_clusters()
# #             print('create')
#             self.create_micro_cluster(X.reshape(1, -1))

    # def find_closest_label_unlabel(self):
    #     min_distance = sys.float_info.max
    #     ucluster = None
    #     lcluster = None
    #     for x in self.micro_clusters:
    #         if x.labelMc == -1:
    #             for y in self.micro_clusters:
    #                 if y.labelMc != -1:
    #                     dist = distance.euclidean(
    #                         x.get_center(), y.get_center())
    #                     if dist < min_distance:
    #                         min_distance = dist
    #                         ucluster = x
    #                         lcluster = y
    #     return lcluster, ucluster

    # def find_closest_label_label(self):
    #     lcluster_1 = None
    #     lcluster_2 = None
    #     sum_class = np.zeros(NumClasses)
    #     dataoflabel = []
    #     min_distance = sys.float_info.max
    #     for x in self.micro_clusters:
    #         sum_class[x.labelMc] += 1
    #     nb_max = np.max(sum_class)
    #     label_index = np.where(sum_class == nb_max)
    #     label = label_index[0][0]
    #     for y in self.micro_clusters:
    #         if y.labelMc == label:
    #             dataoflabel.append(y)
    #     length = len(dataoflabel)
    #     for i, cluster in enumerate(dataoflabel):
    #         center = cluster.get_center()
    #         if ((i+1) < length):
    #             for next_cluster in dataoflabel[i+1:]:
    #                 dist = distance.euclidean(
    #                     center, next_cluster.get_center())
    #                 if dist < min_distance:
    #                     min_distance = dist
    #                     lcluster_1 = cluster
    #                     lcluster_2 = next_cluster
    #     assert(lcluster_1 != lcluster_2)
    #     return lcluster_1, lcluster_2

    # def predict(self, X=None):
    #     """Predict the class labels for the provided data
    #     Parameters
    #     ----------
    #     X :
    #     Returns
    #     -------
    #     y :
    #     """
    #     cluster_centers = list(
    #         map((lambda i: i.get_center()), self.micro_clusters))
    #     #centers_weights = list(map((lambda i: i.get_weight()), self.micro_clusters))
    #     kmeans = KMeans(n_clusters=self.nb_macro_cluster, random_state=1)
    #     result = kmeans.fit_predict(X=cluster_centers, y=None)
    #     return result, kmeans.cluster_centers_

    def get_MC_centers(self):
        return list((x.get_center()) for x in self.micro_clusters)
