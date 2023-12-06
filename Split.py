import sklearn
from sklearn.model_selection import train_test_split

def split_data(data, train_size):
    x_train, x_test = train_test_split(data, train_size=train_size)

    return x_train, x_test

