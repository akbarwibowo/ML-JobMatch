import numpy as np

import One_Hot
import Read_Data
import Tokenizer
import Pad_Sequence
import Split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import csv
import sklearn
from sklearn.model_selection import train_test_split

from Read_Data import read_file
from One_Hot import to_one_hot
from Tokenizer import tokenizer
from Pad_Sequence import padded
from Split import split_data

# here is the flow
'''
1. define the path of the CSV file
2. define the column name wish to extract, store it in variable
3. split to train and test
4. one-hot encode the label
5. tokenize each training feature
6. pad sequence both training and test for each feature. when call padded function for test data, pass tokenizer for training
7. done, data ready to be fed
'''

path = './Dataset/data_capstone.csv'
split_size = 0.9
num_words = 200
maxlen = 425
vocab_size = 300

data = []
keys = ['job_title', 'degree', 'key_skills', 'job_experience']

with open('./Dataset/data_capstone.csv', newline='') as f:
    read = csv.DictReader(f, delimiter=';')
    for row in read:
        item = {}
        for key in keys:
            item[key] = row[key].strip()
        data.append(item)

train, test = train_test_split(data, test_size=0.2)

x_train = np.array([data['degree']+ " " + data['key_skills']+ " " + data['job_experience'] for data in train])
x_test = np.array([data['degree'] + data['key_skills'] + data['job_experience'] for data in test])
print(x_train)
# y_train = [i['job_title'] for i in train]
# y_test = [i['job_title'] for i in test]
#
# all_y = y_train + y_test
# all_y = np.array(all_y)
#
# num = len(all_y)
#
# y_numerate = {label: i for i, en}
