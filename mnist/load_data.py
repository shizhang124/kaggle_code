#coding:utf-8
import csv
import numpy as np
class Data():
    def __init__(self):
        self.train_pic_path = '../../data/mnist/train.csv'
        self.test_pic_path = '../../data/mnist/test.csv'
        self.result_path = '../../data/mnist/result.csv'

    def load_csv_file(self, file_path):
        return np.loadtxt(file_path, dtype=np.str, delimiter=',')

    def load_train_data(self):
        tmp = self.load_csv_file(self.train_pic_path)
        data = tmp[1:, 1:].astype(np.uint8)
        label = tmp[1:, 0].astype(np.uint8)
        return data, label

    def load_test_data(self):
        tmp = self.load_csv_file(self.test_pic_path)
        data = tmp[1:, :].astype(np.uint8)
        return data

    def wirte_result(self, results, path = ''):
        if path == '':
            path = self.result_path
        results = np.asarray(results)
        np.savetxt(path, results, fmt='%d', delimiter=',', header="ImageId,Label", comments='')

