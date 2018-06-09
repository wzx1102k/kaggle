import os
import sys
import utils

'''
@:param
cur_label : current node {attr:label}
branch: sub Node list
split_attr: index of best attr to split sub sample
split_label: split attr label dict list.
eg. [{1:1}, {0:1}, {-1:0.5}]  -1 used for attr unlabeled sample, 0.5 used for unlabeled sample weight
leaf_value: sample index list if node is leaf
leaf_label: sample label if node is leaf
'''

class Node(object):
    def __init__(self, branch={}, split_attr=None, split_label={}, leaf_value=[], leaf_label=None):
        self.branch = branch
        self.split_attr = split_attr
        self.split_label = split_label
        self.leaf_value = leaf_value
        self.leaf_label = leaf_label

    def getLabel(self, sample):
        if self.leaf_label != None:
            return self.leaf_label
        if sample[self.split_attr] != -1:
            sub_branch = self.branch[sample[self.split_attr]]
            return sub_branch.getLabel(sample)
        else:
            return None
            #TBD how to get label of unvalue sample
            '''
            label_weight = self.getUnvalidLabel(sample)
            max_label = None
            max_value = 0
            for key in label_weight:
                if label_weight[key] > max_value:
                    max_label = key
                    max_value = label_weight[key]
            return max_label
    
            def getL
            '''