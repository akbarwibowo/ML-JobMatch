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
from One_Hot import all_one_hot
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
vocab_size = 425
padding = 'post'
truncating = 'post'
embedding_dim = 64

job_title = read_file(path, 'job_title', True)
key_skill = read_file(path, 'key_skills', True)
degree = read_file(path, 'degree', True)
job_exp = read_file(path, 'job_experience', True)

x = []

for i in range(len(key_skill)):
    key = key_skill[i]
    deg = degree[i]
    job = job_exp[i]
    x.append(key + deg + job)
x = np.array(x)

x_train, x_val = split_data(x, split_size)

x_token = tokenizer(x_train, 'oov')

x_train_pad = padded(x_token, x_train, padding, truncating, maxlen)
x_val_pad = padded(x_token, x_val, padding, truncating, maxlen)

y_train, y_val = split_data(job_title, split_size)

y_train_hot = all_one_hot(y_train)
y_val_hot = all_one_hot(y_val)

print(x_train_pad.shape, x_val_pad.shape, y_train_hot.shape, y_val_hot.shape)
print(x_train_pad)

