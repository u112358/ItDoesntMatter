import sys
import os
import numpy as np
sys.path.append('/usr/local/lib/python3.6/site-packages')

import _util

from sklearn.naive_bayes import MultinomialNB

_POSITIVE_CSV_PATH_ = './outputs/positive.csv'
_NEGATIVE_CSV_PATH_ = './outputs/negative.csv'


class Analyse_nb(object):
    def __init__(self,positivez_path, negative_path):

        self.positivez_path = positivez_path
        self.negative_path = negative_path
        self.train_data_p = []
        self.train_tag_p = []
        self.train_data_n = []
        self.train_tag_n = []

        self.read_data()

    def read_data(self):
        with open(self.positivez_path, 'r') as file:
            for line in file: 
                self.train_data_p.append(line.split(', '))
        
        self.train_data_p = np.asarray(self.train_data_p)

        with open(self.negative_path, 'r') as file:
            for line in file:
                self.train_data_n.append(line.split(', '))

        self.train_data_n = np.asarray(self.train_data_n)

        print(self.train_data_p, self.train_data_n)


test = Analyse_nb(_POSITIVE_CSV_PATH_, _NEGATIVE_CSV_PATH_)
