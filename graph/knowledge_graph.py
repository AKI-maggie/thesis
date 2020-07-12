# This file contains any class that provides a knowledge-graph
# framework service

import numpy as np
import os
from data.concept_net import ConceptNet
import tensorflow as tf
from data.progress_bar import ProgressBar
import math

class KnowledgeGraph():
    def __init__(self, nodes, search_engine):
        self.nodes = nodes
        self.search_engine = search_engine
        self.edges = {}
        self.prob_matrix = np.zeros([len(nodes), len(nodes)])
        self.simi_matrix = np.zeros([len(nodes), len(nodes)])

        # progress bar
        self.progress = ProgressBar()

    def _register_neighbors(self):
        pass

    def get_similarity(self):
        pass

# ConceptNet-based knowledgegraph framework
class CN_based_KnowledgeGraph(KnowledgeGraph):
    def __init__(self, nodes, restart_rate, max_iter, save_path):
        print("### Use the ConceptNet-based engine to build the knowledge graph ###")
        super().__init__(nodes, ConceptNet())
        self.restart_rate = restart_rate
        self.max_iter = max_iter

        # check if can directly load the pre-trained similarity matrix
        if os.path.isfile(save_path):
            print ("Pre-trained graph exists, directly loading...")
            self.simi_matrix = np.loadtxt(save_path)
            for i in range(self.simi_matrix.shape[0]):
                rank_idx = self.simi_matrix[i].argsort()[-3:][::-1]
                print("The closest words for {0} are: {1}, {2}, {3}"\
                    .format(self.nodes[i], self.nodes[rank_idx[0]],\
                            self.nodes[rank_idx[1]], self.nodes[rank_idx[2]]))
                print(self.simi_matrix[i])
        else:
            print ("Pre-trained Graph not exist, start training...")
            self._build_edges()
            self._build_prob_matrix()
            self._build_simi_matrix(self.restart_rate, self.max_iter)
            np.savetxt(save_path, self.simi_matrix)
        
        self.simi_tf = tf.convert_to_tensor(self.simi_matrix)

    def _register_neighbors(self, w, neighbor):
        if w in self.edges.keys():
            self.edges[w].append(neighbor)
        else:
            self.edges[w] = [neighbor]

    def _build_prob_matrix(self):
        for i in range(len(self.nodes)):
            if self.nodes[i] not in self.edges.keys():
                continue
            print("Class {0} has {1} neighbors"\
                  .format(self.nodes[i], len(self.edges[self.nodes[i]])))
            print(self.edges[self.nodes[i]])
            prob = 1/float(len(self.edges[self.nodes[i]]))
            for each in self.edges[self.nodes[i]]:
                j = self.nodes.index(each)
                self.prob_matrix[i][j] = prob

    def _build_edges(self):
        self.progress.update_progress(0)
        for i in range(1, len(self.nodes)-1):
            for j in range(i+1, len(self.nodes)):
                e = self.search_engine.check_edge(self.nodes[i], self.nodes[j])
                if e:
                    self._register_neighbors(self.nodes[i], self.nodes[j])
                    self._register_neighbors(self.nodes[j], self.nodes[i])
            self.progress.update_progress(i/(len(self.nodes)-1))
        self.progress.update_progress((len(self.nodes)-1)/(len(self.nodes)-1))
    
    def _build_simi_matrix(self, restart_rate, max_iter=20):
        # first build Rs and R's
        r_matrix = np.zeros([len(self.nodes), len(self.nodes)])
        for i in range(self.prob_matrix.shape[0]):
            # set the starting vector
            start = np.zeros(self.prob_matrix.shape[1])
            start[i] = 1
            v = start.copy()
            # random walk
            for step in range(max_iter):
                v = (1-restart_rate)*np.dot(self.prob_matrix, v) + restart_rate * start
            r_matrix[i] = v    
        
        # then build simi_matrix
        for i in range(self.prob_matrix.shape[0]):
            for j in range(i, self.prob_matrix.shape[0]):
                r_ij = r_matrix[i][j]
                r_ji = r_matrix[j][i]
                s = math.sqrt(r_ij * r_ji)
                self.simi_matrix[i][j] = s
                self.simi_matrix[j][i] = s
        
        for i in range(self.simi_matrix.shape[0]):
            rank_idx = self.simi_matrix[i].argsort()[-3:][::-1]
            print("The closest words for {0} are: {1}, {2}, {3}"\
                  .format(self.nodes[i], self.nodes[rank_idx[0]],\
                          self.nodes[rank_idx[1]], self.nodes[rank_idx[2]]))
            print(self.simi_matrix[i])

    def get_similarity(self):
        return self.simi_tf