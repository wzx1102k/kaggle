import os
import sys
import utils
from node import Node
from tree import DTree
from samples import SampleClass


outlook = {'sunny':0, 'overcast':1, 'rain':2}
temperature = {'hot':0, 'mild':1, 'cool':2}
humidity = {'high':0, 'normal':1}
windy = {'false':0, 'true':1}
play = {'yes':1, 'no':0}

attr = [outlook, temperature, humidity, windy, play]

samples = [
    ['sunny',    'hot',  'high',    'false',    'no'],
    ['sunny',    'hot',  'high',    'true',     'no'],
    ['overcast', 'hot',  'high',    'false',    'yes'],
    ['rain',     'mild', 'high',    'false',    'yes'],
    ['rain',     'cool', 'normal',  'false',    'yes'],
    ['rain',     'cool', 'normal',  'true',     'no'],
    ['overcast', 'cool', 'normal',  'true',     'yes'],
    ['sunny',    'mild', 'high',    'false',    'no'],
    ['sunny',    'cool', 'normal',  'false',    'yes'],
    ['rain',     'mild', 'normal',  'false',    'yes'],
    ['sunny',    'mild', 'normal',  'true',     'yes'],
    ['overcast', 'mild', 'high',    'true',     'yes'],
    ['overcast', 'hot',  'normal',  'false',    'yes'],
    ['rain',     'mild', 'high',    'true',     'no']]

if __name__ == '__main__':
    _samples = []
    for sample in samples:
        _samples.append([attr[i][sample[i]] for i in range(0, len(sample))])

    dtree = DTree()
    sample_class = SampleClass(samples=_samples, calSplitAttrFunc=utils.calMaxEntropy)
    tree = dtree.buildTree(sample_class)
    dtree.printTree(tree)



