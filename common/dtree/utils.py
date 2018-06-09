import sys
import os
import numpy as np


def getMaxLabel(samples, bPackage=False):
    label_list = getLabelCnt(samples, bPackage=bPackage)
    max_label = 0
    max_cnt = 0
    for key in label_list:
        if label_list[key] > max_cnt:
            max_label = key
            max_cnt = label_list[key]
    return max_label

def getLabelCnt(samples, bPackage=False):
    if bPackage == True:
        return getColomCnt(samples, samples[0][-1], True)
    else:
        return getColomCnt(samples, -1, False)


def getColomCnt(samples, attr, bPackage=False):
    result = {}
    _samples = np.array(samples, dtype='uint8')
    if _samples.shape[0] == 0 or _samples.shape[1] == 0:
        return None
    if bPackage == True:
        _idx = int(np.where(_samples[0] == attr)[0]) - 1
        _validsamples = _samples[1:, 1:]
    else:
        _idx = attr
        _validsamples = _samples

    attr_set = set(_validsamples[:, _idx])
    for i in attr_set:
        result[i] = (_validsamples[_validsamples[:, _idx] == i, _idx]).size
    return result

def sortSamples(samples, attr, bPackage=False):
    _samples = np.array(samples, dtype='uint8')
    if _samples.shape[0] == 0 or _samples.shape[1] == 0:
        return None
    if bPackage == True:
        _sortidx = int(np.where(_samples[0] == attr)[0])
        _sortsamples = _samples[1:, :]
        _sort = np.vstack((_samples[0, :], _sortsamples[_sortsamples[:, _sortidx].argsort()]))
    else:
        _sort = _samples[_samples[:, attr].argsort()]
    return _sort

def calEntropy(samples, attr, bPackage=False):
    _samples = np.array(samples, dtype='uint8')
    if _samples.shape[0] == 0 or _samples.shape[1] == 0:
        return None
    if bPackage == True:
        _idx = int(np.where(_samples[0] == attr)[0]) - 1
        _validsamples = _samples[1:, 1:]
    else:
        _idx = attr
        _validsamples = _samples

    attr_label = getColomCnt(samples, attr, bPackage)
    entropy = 0
    weight = 1

    if _idx == -1 or _idx == _validsamples.shape[1] - 1:
        for key in attr_label:
            pi = attr_label[key] / _validsamples.shape[0]
            entropy += -pi * np.log2(pi)
    else:
        if -1 not in attr_label:
            unvalid_cnt = 0
        else:
            unvalid_cnt = attr_label[-1]
        for key in attr_label:
            if key != -1:
                _subsamples = _validsamples[_validsamples[:, _idx] == key, :]
                (_entropy,  _weight) = calEntropy(_subsamples, -1)
                _entropy *= _subsamples.shape[0] / (_validsamples.shape[0] - unvalid_cnt)
                entropy += _entropy
        weight = (_validsamples.shape[0] - unvalid_cnt) / _validsamples.shape[0]

    return (entropy, weight)

def calMaxEntropy(samples, bPackage=False):
    _samples = np.array(samples, dtype='uint8')
    if _samples.shape[0] == 0 or _samples.shape[1] == 0:
        return None
    if bPackage == True:
        _validsamples = _samples[1:, 1:]
        _attr_list = _samples[0, 1:]
    else:
        _validsamples = _samples

    result = []
    for i in range(0, _validsamples.shape[1]):
        result.append(calEntropy(_validsamples, i))
    if result[-1] == 0:
        return None, None
    entropys = [(result[-1][0] - x[0])*x[1] for x in result[:-1]]
    _max = max(entropys)
    ### [1.0, 1.0, 1.0]
    if _max == 0.0:
        return None, None
    _idx = entropys.index(_max)
    if bPackage == True:
        _idx = _attr_list[_idx]
    return _max, _idx

if __name__ == '__main__':
    samples =  [[1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [0, 1, -1, 1, 0],
                [1, 1, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, -1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, -1, 1, 1],
                [0, 1, -1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1]]

    samples_p =  [[-1, 0, 1, 2, 3, 4],
                   [0, 1, 1, 1, 0, 1],
                   [1, 1, 0, 1, 0, 1],
                   [2, 0, 1, -1, 1, 0],
                   [3, 1, 1, 1, 0, 1],
                   [4, 0, 1, 0, 1, 0],
                   [5, 1, 0, -1, 1, 1],
                   [6, 0, 0, 0, 0, 0],
                   [7, 1, 0, -1, 1, 1],
                   [8, 0, 1, -1, 0, 0],
                   [9, 0, 0, 0, 1, 0],
                   [10, 0, 1, 1, 1, 1]]

    result = sortSamples(samples, 2)
    print(result)

    result = sortSamples(samples_p, 2, bPackage=True)
    print(result)