import One_Hot
import Read_Data
import Tokenizer
import Pad_Sequence
import Split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

from Read_Data import read_file
from One_Hot import to_one_hot
from Tokenizer import tokenizer
from Pad_Sequence import padded
from Split import split_data

#here is the flow
'''
1. define the path of the CSV file
2. define the column name wish to extract, store it in variable
3. split to train and test, the funtion return x_train and x_test
4. one-hot encode the label
5. tokenize each training feature
6. pad sequence both training and test for each feature. when call padded function for test data, pass tokenizer for training
7. done, data ready to be fed
'''
