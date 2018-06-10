import os
import sys
import utils

'''
@:param
branch: sub Node list
split_attr: index of best attr to split sub sample
split_label: split attr label cnt dict list.
eg. [{1:4}, {0:5}, {-1:-1}]  -1 used for attr unlabeled sample
leaf_value: sample index list if node is leaf
leaf_label: sample label value if node is leaf
label_list: trainig samples label list
'''
class Node(object):
    def __init__(self, branch={}, split_attr=None, split_label={}, leaf_value=[], leaf_label=None, label_list=()):
        self.branch = branch
        self.split_attr = split_attr
        self.split_label = split_label
        self.leaf_value = leaf_value
        self.leaf_label = leaf_label
        self.label_list = label_list

    def getLabel(self, sample):
        if self.leaf_label != None:
            return self.leaf_label
        if sample[self.split_attr] != -1:
            sub_branch = self.branch[sample[self.split_attr]]
            return sub_branch.getLabel(sample)
        else:
            label_weight = self.getUnvalidLabel(sample)
            print('unvalid sample weight:' + str(label_weight))
            max_label = None
            max_value = 0
            for key in label_weight:
                if label_weight[key] > max_value:
                    max_label = key
                    max_value = label_weight[key]
            return max_label


    def getUnvalidLabel(self, sample):
        label = {}
        for key in self.label_list:
            label[key] = 0
        if self.leaf_label != None:
            for key in label:
                if key == self.leaf_label:
                    label[key] = 1
            return label
        else:
            if sample[self.split_attr] != -1:
                sub_branch = self.branch[sample[self.split_attr]]
                return sub_branch.getUnvalidLabel(sample)
            else:
                total_cnt = 0
                for split_key in self.split_label:
                    if split_key != -1:
                        total_cnt += self.split_label[split_key]

                for split_key in self.split_label:
                    if split_key != -1:
                        sub_branch = self.branch[split_key]
                        sub_label = sub_branch.getUnvalidLabel(sample)
                        for label_key in label:
                            label[label_key] += sub_label[label_key] * (self.split_label[split_key] / total_cnt)
                return label
