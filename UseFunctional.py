from os import getcwd

import numpy as np

import One_Hot
import Pad_Sequence
import Read_Data
import Split
import tensorflow as tf
import sklearn
import Tokenizer as tk

from sklearn.utils import shuffle
from One_Hot import all_one_hot
from One_Hot import unique_one_hot
from Pad_Sequence import padded
from Tokenizer import tokenizer
from Read_Data import read_file
from Split import split_data
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# initiate variables
cwd = getcwd()
path = f"{cwd}/Dataset/data_capstone.csv"  # path for the dataset
train_size = 0.9
oov = 'oov'
padding = 'post'
truncating = 'post'
maxlen = 500
vocab_size = 400

# get each column
job_title = read_file(path, 'job_title', True)
degree = read_file(path, 'degree', True)
key_skill = read_file(path, 'key_skills', True)
job_experience = read_file(path, 'job_experience', True)

# split all retrieved data to train and test
key_skill_train, key_skill_test = split_data(key_skill, train_size)
degree_train, degree_test = split_data(degree, train_size)
job_experience_train, job_experience_test = split_data(job_experience, train_size)
job_title_train, job_title_test = split_data(job_title, train_size)

job_title_train_hot = all_one_hot(job_title_train)
job_title_test_hot = all_one_hot(job_title_test)

# option 1 for degree: tokenized
degree_token = tokenizer(degree_train, oov, vocab_size)
degree_train_pad = padded(degree_token, degree_train, padding, truncating, maxlen)  # train ready data
degree_test_pad = padded(degree_token, degree_test, padding, truncating, maxlen)  # test ready data

# option 2 for degree: unique one-hot
# there are only three categorical. print it to see it
degree_train_hot_unique = unique_one_hot(degree_train)
degree_test_hot_unique = unique_one_hot(degree_test)

# option 3 for degree: all one-hot
# each value in one-hot encoded, even it is the same category. print it to see it
degree_train_hot_all = all_one_hot(degree_train)
degree_test_hot_all = all_one_hot(degree_test)

# option 1 for job_experience: tokenized
job_experience_token = tokenizer(job_experience_train, oov, vocab_size)
job_experience_train_pad = padded(job_experience_token, job_experience_train, padding, truncating, maxlen)  # train ready data
job_experience_test_pad = padded(job_experience_token, job_experience_test, padding, truncating, maxlen)  # test ready data

# option 2 for job_experience: one-hot (since this data is numerical, only as an option)
job_experience_train_hot = all_one_hot(job_experience_train)
job_experience_test_hot = all_one_hot(job_experience_test)

# input for key_skill
key_skill_token = tokenizer(key_skill_train, oov, vocab_size)
key_skill_train_pad = padded(key_skill_token, key_skill_train, padding, truncating, maxlen)  # key_skill 1 train
key_skill_test_pad = padded(key_skill_token, key_skill_test, padding, truncating, maxlen)  # key_skill 1 test

key_skill_2_train_pad, key_skill_2_test_pad = shuffle(key_skill_train_pad), shuffle(key_skill_test_pad)  # key_skill 2 train, key_skill 2 test
key_skill_3_train_pad, key_skill_3_test_pad = shuffle(key_skill_train_pad), shuffle(key_skill_test_pad)  # key_skill 3 train, key_skill 3 test
key_skill_4_train_pad, key_skill_4_test_pad = shuffle(key_skill_train_pad), shuffle(key_skill_test_pad)  # key_skill 4 train, key_skill 4 test
key_skill_5_train_pad, key_skill_5_test_pad = shuffle(key_skill_train_pad), shuffle(key_skill_test_pad)  # key_skill 5 train, key_skill 5 test

degree_input = keras.Input(
    shape=(None,),
    name='degree'
)

job_input = keras.Input(
    shape=(None,),
    name='job'
)

key_1_input = keras.Input(
    shape=(None,),
    name='key_1'
)

key_2_input = keras.Input(
    shape=(None,),
    name='key_2'
)

key_3_input = keras.Input(
    shape=(None,),
    name='key_3'
)

key_4_input = keras.Input(
    shape=(None,),
    name='key_4'
)

key_5_input = keras.Input(
    shape=(None,),
    name='key_5'
)

degree_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(degree_input)

job_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(job_input)

key_len = len(key_skill_token.word_index) + 1

key_1_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(key_1_input)

key_2_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(key_2_input)

key_3_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(key_3_input)

key_4_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(key_4_input)

key_5_feature = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=maxlen
)(key_5_input)

degree_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(degree_feature)

job_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(job_feature)

key_1_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(key_1_feature)

key_2_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(key_2_feature)

key_3_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(key_3_feature)

key_4_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(key_4_feature)

key_5_feature = keras.layers.Bidirectional(keras.layers.LSTM(
    128
))(key_5_feature)

degree_feature = keras.layers.Flatten()(degree_feature)
job_feature = keras.layers.Flatten()(job_feature)
key_1_feature = keras.layers.Flatten()(key_1_feature)
key_2_feature = keras.layers.Flatten()(key_2_feature)
key_3_feature = keras.layers.Flatten()(key_3_feature)
key_4_feature = keras.layers.Flatten()(key_4_feature)
key_5_feature = keras.layers.Flatten()(key_5_feature)

x = keras.layers.concatenate([
    degree_feature,
    job_feature,
    key_1_feature,
])

job_title_pred = keras.layers.Dense(3, 'softmax', name='job_output')(x)

model = keras.Model(
    inputs=[degree_input, job_input, key_1_input],
    outputs=job_title_pred
)

model.summary()

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
job_title_train_hot = np.array(job_title_train_hot)
job_title_test_hot = np.array(job_title_test_hot)

job_title_token = Tokenizer()
job_title_token.fit_on_texts(job_title)
job_title_train_seq = job_title_token.texts_to_sequences(job_title_train)
job_title_test_seq = job_title_token.texts_to_sequences(job_title_test)

job_title_train_seq = np.array(job_title_train_seq)
job_title_test_seq = np.array(job_title_test_seq)

print(job_title_train_seq.shape)

# model.fit(
#     {
#         'degree': degree_train_pad,
#         'job': job_experience_train_pad,
#         'key_1': key_skill_train_pad,
#     },
#     {
#         'job_output': job_title_train_hot
#     },
#     32,
#     100
# )
