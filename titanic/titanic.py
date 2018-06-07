import numpy as np
import os
import pandas as pd
from math import isnan

np.set_printoptions(suppress=True)  ## don't output like format : 1.0e+1
training_path = './input/train.csv'
testing_path = './input/test.csv'
batch_rate = 0.1
epoisd = 1000

training_df = pd.read_csv(training_path)
testing_df = pd.read_csv(testing_path)

class passengers(object):
    def __init__(self):
        self.passno, self.survival, self.pclass, self.name, self.sex, \
            self.age, self.sibsp, self.parch, self.ticket, self.fare, \
            self.cabin, self.embarked = self.get_batch(training_df, batch_rate)

    def get_batch(self, reader, rate):
        batch = reader.sample(frac=rate)
        passno = batch.iloc[:, 0].tolist()
        survival = batch.iloc[:, 1].tolist()
        pclass = batch.iloc[:, 2].tolist()
        name = batch.iloc[:, 3].tolist()
        sex = batch.iloc[:, 4].tolist()
        age = batch.iloc[:, 5].tolist()
        sibsp = batch.iloc[:, 6].tolist()
        parch = batch.iloc[:, 7].tolist()
        ticket = batch.iloc[:, 8].tolist()
        fare = batch.iloc[:, 9].tolist()
        cabin = batch.iloc[:, 10].tolist()
        embarked = batch.iloc[:, 11].tolist()
        return passno, survival, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked

    def cal_purity(self, input, label):
        total = input.size
        if total == 0:
            return 0
        zero_size = input[label==0].size
        sum = 1 - (zero_size/total)**2 - ((total-zero_size)/total)**2
        return sum

    def feature_generate(self, no, input, label):
        _input = np.asarray(input)
        para = np.asarray([no, input, label])
        para = para[:, para[1, :].argsort()]
        #print(para)
        gini = 1
        min_index = 0
        valid_cnt = 0
        row, col = para.shape
        feature = np.zeros((col))
        print(row, col)
        for i in range(0, col+1):
            total = self.cal_purity(para[1, 0:i], para[2, 0:i]) + self.cal_purity(para[1, i:col], para[2, i:col])
            print(i, total)
            if total < gini:
                gini = total
                min_index = i
            if i == col or isnan(para[1, i]):
                valid_cnt = i
                print(valid_cnt)
                break
        gini *= valid_cnt/col
        print(min_index)
        print(para[1,:])
        feature[valid_cnt:] = -1
        feature[min_index:valid_cnt] = 1
        para = np.vstack((para, feature))
        print(para)
        #  < para[1, min_index] to split
        return feature, gini

    #list1 ['age', 'sex' ...]
    #list2 [list_age, list_sex]

    def max_purity(self, cal_name, cal_data, cal_label):
        pass




def tree_generate(features, labels):
    return tree

def feature_generate(data, label):

    return feature

def train():
    for i in range(0, epoisd):
        paras, labels = get_batch(training_df, batch_rate)
        features = feature_generate(paras, labels)
        tree = tree_generate(features, labels)

    pass

def predict():
    pass


if __name__ == "__main__":
    samplePassengers = passengers()
    #gini, feature = samplePassengers.feature_generate(samplePassengers.passno, samplePassengers.age, samplePassengers.survival)
    gini, feature = samplePassengers.feature_generate([1,2,3,4,5,6,7,8,9,10,11,12], [5,10,20,21,23,29,31,38,40,42,45,50], [0,0,0,0,0,0,1,1,1,1,0,0])
    #print(samplePassengers.age)
    print(gini, feature)