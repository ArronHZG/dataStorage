import csv
from time import sleep

import numpy as np
import random
import pandas as pd
import threading

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():
    def get_memory(self,train_test):
        data_set = pd.read_csv(f"data/{train_test}_data.csv").values# 加载数据集
        X = data_set[:, 0:10].astype(np.int16)  # 分割为10个输入变量
        Y = data_set[:, 10].astype(np.int16)
        return X,Y

    @threadsafe_generator
    def generator(self, batch_size, train_test):

        data_set = pd.read_csv(f"data/{train_test}_data.csv").values# 加载数据集
        X = data_set[:, 0:10].astype(np.int16)  # 分割为10个输入变量
        Y = data_set[:, 10].astype(np.int16)

        print("Creating %s generator with %d samples." % (train_test, len(data_set)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                sample = random.choice(data_set)
                X.append(sample[0:10].astype(np.int16))
                y.append(sample[10].astype(np.int16))
            yield np.array(X), np.array(y)


if __name__ == '__main__':
    ds=DataSet()
    for x,y in ds.generator(32,'train'):
        print(x)
        print(y)
        sleep(5)
