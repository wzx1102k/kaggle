import os
import sys
from node import Node
import utils
from samples import SampleClass


class DTree(object):
    def __init__(self):
        pass

    def buildTree(self, sample_class):
        if sample_class.samples.shape[0] == 0 or sample_class.samples.shape[1] == 0:
            return None

        sub_class = sample_class.splitSamples()

        if sample_class.split_attr == None:
            sample_label = utils.getMaxLabel(sample_class.samples, bPackage=True)
            return Node(leaf_value=list(sample_class.samples[1:, 0]), leaf_label=sample_label)
        else:
            branch = {}
            for key in sample_class.split_label:
                if key != -1:
                    branch[key] = self.buildTree(sub_class[key])
            return Node(branch=branch, split_attr=sample_class.split_attr, split_label=sample_class.split_label)

    def printTree(self, tree, indent='   '):
        if tree.leaf_value != []:
            print(indent + str(tree.leaf_value))
        else:
            for i in tree.branch:
                print(indent + str(tree.split_attr) + ":" + str(i))
                self.printTree(tree.branch[i], indent+'   ')

if __name__ == "__main__":
    samples =  [[1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1]]
    dtree = DTree()
    sample_class = SampleClass(samples=samples, calSplitAttrFunc=utils.calMaxEntropy)
    tree = dtree.buildTree(sample_class)
    dtree.printTree(tree)