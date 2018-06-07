import os
import numpy as np
import sys

class Node(object):
    def __init__(self, branch=[], attr_idx=None, attr_label=[], sample_value=[], sample_label={}):
        self.branch = branch
        self.attr_idx = attr_idx
        self.attr_label = attr_label
        self.sample_value = sample_value
        self.sample_label = sample_label

    def get_label(self):
        pass


class Id3(object):
    def __init__(self):
        pass

    def getLabelCnt(self, samples):
        return self.getColomCnt(samples, -1)

    def getColomCnt(self, samples, col):
        result = {}
        for rows in samples:
            r = rows[col]
            if r not in result:
                result[r] = 0
            result[r] += 1
        return result

    def calEntropy(self, samples, col):
        rows = len(samples)
        cols = len(samples[0])
        if rows == 0:
            return None
        _count = self.getColomCnt(samples, col)
        entropy = 0
        if col == -1 or col == cols -1:
            for i in _count:
                pi = _count[i]/rows
                entropy += -pi*np.log2(pi)
        else:
            for key in _count:
                sub = [x for x in samples if x[col] == key]
                entropy += self.calEntropy(sub, -1) * len(sub)/rows
        return entropy

    def calMaxEntropy(self, samples):
        if len(samples) == 0:
            return None, None
        result = []
        for i in range(0, len(samples[0])):
            result.append(self.calEntropy(samples, i))
        if result[-1] == 0:
            return None, None
        entropys = [result[-1] - x for x in result[:-1]]
        #print(entropys)
        _max = max(entropys)
        ### [1.0, 1.0, 1.0]
        if _max == 0.0:
            return None, None
        return _max, entropys.index(_max)

    '''
        @:return
        samples: data input
        attr_idx_list: attr index list
        sample_idx_list: samples number list
        
    '''
    def split_samples(self, samples, attr_idx_list, sample_idx_list):
        if len(samples) == 0:
            return None, None, None, None, None

        _attr_idx_list = list(attr_idx_list)
        entropy, idx = self.calMaxEntropy(samples)
        #print(entropy, idx)
        if idx == None:
            return None, None, None, None, None
        attr_idx = _attr_idx_list[idx]
        attr_label = []
        sub_samples = []
        sub_samples_idx = []
        for i in range(0, len(samples)):
            sample = samples[i]
            sample_idx = sample_idx_list[i]
            _sample = list(sample)
            attr = sample[idx]
            if attr not in attr_label:
                attr_label.append(attr)
                sub_samples.append([])
                sub_samples_idx.append([])
            _sample.pop(idx)
            sub_samples[attr_label.index(attr)].append(_sample)
            sub_samples_idx[attr_label.index(attr)].append(sample_idx)
        _attr_idx_list.pop(idx)
        return attr_idx, attr_label, sub_samples, _attr_idx_list, sub_samples_idx

    def buildTree(self, samples, attr_list, sample_list):
        if len(samples) == 0:
            return None
        attr_idx, attr_label, sub_samples, _attr_list, _sample_list = self.split_samples(samples, attr_list, sample_list)
        print(attr_idx)
        #print(sub_samples)
        #print('-------------------')
        #print(_sample_list)
        if attr_idx == None:
            print('**************')
            print(_sample_list)
            #print(label_cnt)
            label_cnt = self.getLabelCnt(samples)
            #print(label_cnt)
            return Node(sample_value=[x for x in _sample_list], sample_label=label_cnt)
        else:
            branch = []
            for i in range(0, len(sub_samples)):
                sub_sample = sub_samples[i]
                sub_sample_list = _sample_list[i]
                #print(sub_sample_list)
                #print(sub_sample)
                branch.append(self.buildTree(sub_sample, _attr_list, sub_sample_list))
            return Node(branch=branch, attr_idx=attr_idx, attr_label=attr_label)

    def printTree(self, tree, indent='   '):
        if tree.sample_value != []:
            print(tree.sample_value)
        else:
            for i in tree.branch:
                #print(indent + str(tree.attr_idx) + ":" + str(i.attr_label))
                self.printTree(i, indent+'   ')

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
    id3 = Id3()
    #_max, _idx = id3.calMaxEntropy(samples)
    rows = [x for x in range(0, len(samples))]
    cols = [x for x in range(0, len(samples[0]))]
    node = id3.buildTree(samples, cols, rows)
    id3.printTree(node)
    #print(_max, _idx)
