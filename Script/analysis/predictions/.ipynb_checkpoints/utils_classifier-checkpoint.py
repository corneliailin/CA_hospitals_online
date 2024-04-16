import pandas as pd
import numpy as np
import os
import glob
import re
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
from embeddings import utils_dt_prep
from scipy.optimize import fsolve
from math import exp
tf.keras.backend.set_floatx('float64')
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    MAX_LEN = 40
    LR = 3e-4#3e-4#0.01 #0.001
    LR_decay = 0.01
    FF_DIM = 564 #64
    FF_LL_DIM = 0 # dimension of last layer
    DROP_RATE = 0.3#0.1#0.3 # dropout rate
    
config = Config()

# initial weight
################
def initial_weights(vectorize_layer, df, y_union, y_tokenized_cols):
    ''''''
    # keep only diag in y_union
    df_union = df[df.diag00_2d.isin(y_union)]

    # add tokenized diag
    df_union['diag00_2d_tokenized'] = utils_dt_prep.encode(vectorize_layer['diag'], df_union.diag00_2d)[:,0]

    # calculate class weights
    d = dict(df_union[['diag00_2d_tokenized']].value_counts())
    new_d = {}
    for key in d.keys():
        new_d[key[0]] = d[key] # rename keys

    d_sorted = {i: new_d[i] for i in list(y_tokenized_cols)} # sort keys as in y_train_tok_cols

    m = np.mean(list(d_sorted.values()))
    class_weight = {k:m/v for (k,v) in d_sorted.items()} 

    # define classes frequency list
    frequency = list(list(d_sorted.values())/sum(d_sorted.values()))

    # define equations to solve initial bias
    def eqn(x, frequency=frequency):
      sum_exp = sum([exp(x_i) for x_i in x])
      return [exp(x[i])/sum_exp - frequency[i] for i in range(len(frequency))]

    # calculate init bias
    bias_init = fsolve(func=eqn,
                    x0=[0]*len(frequency),
                    ).tolist()
    

    return bias_init, class_weight


# classifier model
##################
def classifier_model(pretrained_bert_model, FF_LL_DIM, keys, initial_bias=None):
    ''''''
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    # define number of classess to predict
    config.FF_LL_DIM = FF_LL_DIM
     
    # add initial bias
    if initial_bias is not None:
        initial_bias = tf.keras.initializers.Constant(initial_bias)
    
    # define input layers (follows the MLM structure)
    input_list = []
    inputs_diag = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='diag')
    input_list.append(inputs_diag)
    
    # if only age added extra        
    if len(keys)==2 and keys[1]=='age':
        inputs_age = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='age')
        input_list.append(inputs_age)

    # if only cnty added extra
    if len(keys)==2 and keys[1]=='cnty':
        inputs_cnty = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='cnty')
        input_list.append(inputs_cnty)

    # if both cnty and age added extra
    if len(keys)==3:
        inputs_age = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='age')
        input_list.append(inputs_age)
        
        inputs_cnty = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='cnty')
        input_list.append(inputs_cnty)
    
    # define classification model layers
    sequence_output = pretrained_bert_model(input_list)  # this layer provides the token embeddings
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output) # sentence embeddings
    hidden_layer = layers.Dense(config.FF_DIM, activation="relu")(pooled_output)
    dropout = layers.Dropout(config.DROP_RATE)(hidden_layer)
    classifier_outputs = layers.Dense(
        config.FF_LL_DIM,
        activation="softmax",
        bias_initializer=initial_bias,
    )(dropout) 
    
    # create model
    classifier_model = keras.Model(
        inputs = input_list,
        outputs = classifier_outputs,
        name="classification_nodel")
    
    # compile model
    classifier_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = config.LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        weighted_metrics=[],
        metrics=["accuracy"]
    )
    
    return classifier_model

