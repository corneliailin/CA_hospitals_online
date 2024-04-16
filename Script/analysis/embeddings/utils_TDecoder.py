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
tf.keras.backend.set_floatx('float64')
import warnings
warnings.filterwarnings('ignore')

## set-up configuration
#######################
@dataclass
class Config:
    MAX_LEN =40
    LR = 5e-5
    LR_decay = 0.01
    EMBED_DIM = 128
    NUM_HEAD = 12 
    FF_DIM = 128 
    NUM_LAYERS = 6
    RATE = 0.1 # dropout rate 
    FF_LL_DIM = 0 # last layer dimension
config = Config()


# Optimizer
###########
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float64)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(config.EMBED_DIM)


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



## create masked language bert model ##
#######################################
def create_TDecoder(vectorize_layer, sample_weights, keys, FF_LL_DIM):
    '''
    '''
    config.FF_LL_DIM = FF_LL_DIM
    
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

    ## DECODER ##
    #############
    # The decoder receives the embeddings
    decoder_output = embeddings
    for i in range(config.NUM_LAYERS): 
        
        ## Multi-Head self-attention ##
        ###############################
        query =  decoder_output
        key = decoder_output
        value = decoder_output

        # add multihead attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=config.NUM_HEAD,  # implement 8 attention layers
            key_dim=config.EMBED_DIM // config.NUM_HEAD,
            name="encoder_{}/att".format(i),
        )(query, key, value)

        # add droput layer
        attention_output = layers.Dropout(
            0.1, name="encoder_{}/att_dropout".format(i))(
            attention_output
        )

        # add normalization layer
        attention_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
        )(query + attention_output)

        ## Feed Forward layer ##
        ########################
        # fully connected layer
        ffn = keras.Sequential(
            [layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM)],
            name="encoder_{}/ffn".format(i),
        )
        ffn_output = ffn(attention_output)
        # add dropout layer
        ffn_output = layers.Dropout(
            config.RATE, name="encoder_{}/ffn_dropout".format(i))(
            ffn_output
        )
        
        # add normalization layer
        sequence_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
        )(attention_output + ffn_output)

        
        # pooled outout
        decoder_output = sequence_output
    decoder_output = layers.GlobalMaxPooling1D()(decoder_output)

    # includes the multi-head attention, and fully connected layer
    TDecoder_output = layers.Dense(
        config.FF_LL_DIM,
        name="mlm_cls",
        activation="softmax")(decoder_output)
    
    # create model 
    TDecoder_model = keras.Model(
        inputs = input_list,
        outputs = TDecoder_output,
        name="TDecoder_nodel")
    
    
    ## Compile ##
    #############
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    

    # compile model
    TDecoder_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        weighted_metrics=[],
        metrics=["accuracy"]
    )

    return TDecoder_model
