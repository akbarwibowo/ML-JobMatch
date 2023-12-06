import tensorflow as tf
from  tensorflow import  keras

#func to tokenize sentences
def tokenizer(sentences, num_words, oov_token):
    #make tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, filters="|/", lower=True, oov_token=oov_token)

    #fit tokenizer into sentences
    tokenizer.fit_on_texts(sentences)

    return tokenizer

