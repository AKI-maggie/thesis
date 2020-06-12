# This file contains the class ConceptNet which would provide
# all services related to the public API service of ConceptNet

import requests
import sys
import time

class ConceptNet():
    def __init__(self):
        self.api_addr = 'http://api.conceptnet.io/'
        self.related_query = self.api_addr + 'related/c/en/'
        self.edge_query = self.api_addr + 'query?'

    def check_related(self, w):
        obj = requests.get(self.related_query + w).json()
        return obj

    def check_edge(self, w1, w2, attempts = 0):
        keywords = 'node=/c/en/{0}&other=/c/en/{1}'.format(w1, w2)
        edges = []
        flag = 0
        if attempts in set(range(10)):  # try multiple times if single api usage failed
            try:
                obj = requests.get(self.edge_query + keywords).json()
            except:
                print("ERROR TIME {2}: API usage failed when checking edges between {0} and {1}."\
                  .format(w1, w2, attempts))
                time.sleep(10)
                edges = self.check_edge(w1, w2, attempts + 1)
                flag = 1
        else:
            print("ERROR: API usage killed.")
            sys.exit()

        # preprocess relationships
        if flag == 0:
            for each in obj['edges']:
                if each['surfaceText'] != None:
                    edges.append(each['surfaceText'])
        return edges