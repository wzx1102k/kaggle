import sys
import os
import utils
import numpy as np


class SampleClass(object):
    def __init__(self, samples, split_attr=None, split_label={}, label_list=(), calSplitAttrFunc=None, bPackage=False):
        if bPackage == False:
            _real_samples = np.array(samples)
            _attr_idx = np.linspace(-1, _real_samples.shape[1] - 1,  _real_samples.shape[1] + 1)
            _sample_idx = np.array([np.linspace(-1, _real_samples.shape[0] - 1, _real_samples.shape[0] + 1)]).T
            self.samples = np.vstack((_attr_idx, np.hstack((_sample_idx[1:, ], _real_samples)))).astype('int')
        else:
            self.samples = samples

        if label_list == ():
            self.label_list = tuple(set(self.samples[1:, -1]))
        else:
            self.label_list = label_list
        self.split_attr = split_attr
        self.split_label = split_label
        self.calSplitAttrFunc=calSplitAttrFunc

    '''
    @ return sub sample_class list
    '''
    def splitSamples(self):
        if self.samples.shape[0] == 0 or self.samples.shape[1] == 0:
            return []

        split_value, split_attr = self.calSplitAttrFunc(self.samples, bPackage=True)
        if split_attr == None:
            return []

        self.split_label = utils.getColomCnt(self.samples, split_attr, bPackage=True)
        self.split_attr = split_attr

        _idx = int(np.where(self.samples[0] == split_attr)[0])
        _samples = np.delete(self.samples, _idx, axis=1)
        attr_values = self.samples[1:, _idx]
        _valid_samples = _samples[1:]

        sub_sample_class = {}
        for key in self.split_label:
            if key != -1:
                sub_samples = np.vstack((_samples[0], _valid_samples[attr_values == key, :],
                                         _valid_samples[attr_values == -1, :]))
                sub_sample_class[key] = SampleClass(samples=sub_samples, calSplitAttrFunc=self.calSplitAttrFunc,
                                                    label_list=self.label_list, bPackage=True)
            else:
                sub_sample_class[key] = {}

        return sub_sample_class


if __name__ == '__main__':
    samples = [[1, 1, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 1, -1, 1, 0],
                [1, 1, 1, 0, 1],
                [1, 1, 0, 1, 0],
                [1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1],
                [0, 1, -1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1]]

    sample_class = SampleClass(samples=samples, calSplitAttrFunc=utils.calMaxEntropy)
    sub_class = sample_class.splitSamples()
    print(sample_class.samples)
    print(sample_class.split_attr)
    print(sample_class.split_label)
    print(sample_class.sample_label)
    for key in sample_class.split_label:
        if key != -1:
            print(sub_class[key].samples)
