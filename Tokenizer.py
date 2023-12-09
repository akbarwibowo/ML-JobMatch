import tensorflow as tf
from  tensorflow import  keras

#func to tokenize sentences
def tokenizer(sentences, oov_token, vocab_size):
    #make tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(filters="|/", lower=True, oov_token=oov_token, num_words=vocab_size)

    #fit tokenizer into sentences
    tokenizer.fit_on_texts(sentences)

    return tokenizer

