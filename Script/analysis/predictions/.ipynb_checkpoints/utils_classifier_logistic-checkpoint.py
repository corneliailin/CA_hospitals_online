import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
tf.keras.backend.set_floatx('float64')

import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    MAX_LEN = 40
    LR = 3e-4#3e-4#0.01 #0.001
    LR_decay = 0.01
    FF_DIM = 64
    FF_LL_DIM = 0 # dimension of last layer
    RATE = 0.1 # dropout rate
    NUM_FEATURES = 0
    
config = Config()


# create classifier logistic model
def classifier_logistic_model(NUM_FEATURES, FF_LL_DIM):
    ''''''
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    
    # define number of classess to predict
    config.NUM_FEATURES = NUM_FEATURES
    config.FF_LL_DIM = FF_LL_DIM
    

    # use Keras Sequential API to build a logistic regression
    classifier_logistic_model = keras.Sequential()
    
    # create input layer
    classifier_logistic_model.add(tf.keras.Input(shape=(config.NUM_FEATURES,),
         name='Input'
    ))

    
    classifier_logistic_model.add(keras.layers.Dense(
        units=config.FF_LL_DIM,  
        use_bias=False,
        activation='softmax',
        name="Output"
    ))
  
   # compile
    classifier_logistic_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = config.LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return classifier_logistic_model
