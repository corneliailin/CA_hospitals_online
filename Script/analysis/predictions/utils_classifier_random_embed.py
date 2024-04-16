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
from scipy.optimize import fsolve
from math import exp
tf.keras.backend.set_floatx('float64')
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../')
from embeddings import utils_dt_prep


@dataclass
class Config:
    MAX_LEN = 40
    LR = 3e-4#3e-4#0.01 #0.001
    LR_decay = 0.01
    FF_DIM = 564#64
    FF_LL_DIM = 0 # dimension of last layer
    DROP_RATE = 0.3#0.1#0.3 # dropout rate
    EMBED_DIM = 128 
    VOCAB_SIZE = 124

    
config = Config()


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


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
def classifier_model(vectorize_layer, FF_LL_DIM, keys, initial_bias=None):
    ''''''
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    # define number of classess to predict
    config.FF_LL_DIM = FF_LL_DIM
    
    # add initial bias
    if initial_bias is not None:
        initial_bias = tf.keras.initializers.Constant(initial_bias)
    
    ## input layers ##
    ##################
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
        

    ## embedding layers ##
    #####################
    # diag
    diag_embeddings = layers.Embedding(
        len(vectorize_layer['diag'].get_vocabulary()),
        config.EMBED_DIM,
        name="diag_emb"
    )(inputs_diag)
    
    # pos
    pos_embeddings = layers.Embedding(
        config.MAX_LEN,
        config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="pos_emb"
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    
    
    # if only age added extra
    if len(keys)==2 and keys[1]=='age':  
        # age
        age_embeddings = layers.Embedding(
            #config.MAX_LEN,
            len(vectorize_layer['age'].get_vocabulary()),
            config.EMBED_DIM,
            name="age_emb"
        )(inputs_age)
        

    # if only cnty added extra
    if len(keys)==2 and keys[1]=='cnty':  
        # cnty
        cnty_embeddings = layers.Embedding(
            #config.MAX_LEN,
            len(vectorize_layer['cnty'].get_vocabulary()),
            config.EMBED_DIM,
            name="cnty_emb"
        )(inputs_cnty)
        
    
    # if both cnty and age added extra
    if len(keys)==3:
        # age
        age_embeddings = layers.Embedding(
            #config.MAX_LEN,
            len(vectorize_layer['age'].get_vocabulary()),
            config.EMBED_DIM,
            name="age_emb"
        )(inputs_age)
        
        # cnty
        cnty_embeddings = layers.Embedding(
            #config.MAX_LEN+1,
            len(vectorize_layer['cnty'].get_vocabulary()),
            config.EMBED_DIM,
            name="cnty_emb"
        )(inputs_cnty) 
    
    
    # add embeddings
    if len(keys)==1:
        embeddings = diag_embeddings + pos_embeddings
        
    if len(keys)==2 and keys[1]=='age':
        embeddings = diag_embeddings + pos_embeddings + age_embeddings

    if len(keys)==2 and keys[1]=='cnty':
        embeddings = diag_embeddings + pos_embeddings + cnty_embeddings
        
    if len(keys)==3:
        embeddings = diag_embeddings + pos_embeddings + age_embeddings + cnty_embeddings
    
    # define classification model layers
    #sequence_output = pretrained_bert_model(input_list)  # this layer provides the token embeddings
    sequence_output = embeddings
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output) # sentence embeddings
    hidden_layer = layers.Dense(config.FF_DIM, activation="relu")(pooled_output)
    dropout = layers.Dropout(config.DROP_RATE)(hidden_layer)
    classifier_outputs = layers.Dense(
        config.FF_LL_DIM,
        activation="softmax",
        bias_initializer=initial_bias)(dropout) 
    
    # create model
    classifier_model = keras.Model(
        inputs = input_list,
        outputs = classifier_outputs,
        name="logit_classification_nodel")
    
    # compile model
    classifier_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = config.LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        weighted_metrics=[],
        metrics=["accuracy"]
    )
    
    return classifier_model

