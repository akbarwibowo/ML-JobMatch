import tensorflow as tf
from  tensorflow import  keras

def padded(tokenizer, sentences, padding, truncating, maxlen):
    #make sequences
    sequence = tokenizer.texts_to_sequences(sentences)

    #pad the sequences
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences=sequence, maxlen=maxlen, padding=padding, truncating=truncating)

    return  padded_sequences