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


## define extra variables
###########################
def create_extra_features(df, embeddings_pool):
    ''''''
    df['precare'] = np.where(df.precare.eq('-'), '99',
                                       np.where(df.precare.isna(), '99', df.precare)).astype(int)
    df['prevsts'] = np.where(df.prevsts.isna(), 99, df.prevsts).astype(int)
    
    # one hot encodings #
    #####################
    sex_one_hot = pd.get_dummies(df['sexI'], drop_first=True)
    race_one_hot = (pd.get_dummies(df['raceI'], drop_first=False).drop(['white'], axis=1)) #native american/eskimo/aleut
    #bthyear_one_hot = pd.get_dummies(df['bthyearI'], prefix='bthyear', drop_first=True)
    #cntyresI_one_hot =  pd.get_dummies(df['cntyresI'], prefix='cntyresI',drop_first=True)
    meduc_one_hot =  (pd.get_dummies(df['meduc'], prefix='meduc',drop_first=False).drop(['meduc_masters or phd'], axis=1))
    mrace_one_hot = (pd.get_dummies(df['raceM'], drop_first=False).drop(['white'], axis=1)) #native american/eskimo/aleut

    # compute pm25_hat #
    ####################
    df['pm25I'] = df.pm25I.astype(int)
    df['intercept'] = 1
    
    # y and X
    y = df[['pm25I']]
    X = pd.concat(
        [
            df[['intercept', 'visitsM_9mpp', 'visitsM_1ypp', 'visitsI_1yol',  'precare', 'prevsts']], 
            #df[['intercept', 'wfeI', 'visitsM_9mpp', 'visitsM_1ypp', 'visitsI_1yol']],
            #sex_one_hot, race_one_hot,
            meduc_one_hot, mrace_one_hot
        ],
        axis=1
    ) 
    
    if "unknown" in X.columns:
        X.drop(columns=["unknown"], inplace = True) #remove duplicate columns
    if "9.0" in X.columns:
        X.drop(columns=["9.0"], inplace = True) #remove duplicate columns
        
    X = pd.concat([X, pd.DataFrame(embeddings_pool)], axis=1)

    return X

    # fit model
    fit_1st_stage = sm.OLS(y, X).fit()
    #print(fit_1st_stage.summary())

    # save pm25I_hat
    X['pm25I_hat'] = fit_1st_stage.get_prediction(X).summary_frame()['mean']

    return X


# create classifier plus model
def classifier_plus_model(NUM_FEATURES, FF_LL_DIM):
    ''''''
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    
    # define number of classess to predict
    config.NUM_FEATURES = NUM_FEATURES
    config.FF_LL_DIM = FF_LL_DIM
    

    # use Keras Sequential API to build a logistic regression
    classifier_plus_model = keras.Sequential()
    
    # create input layer
    classifier_plus_model.add(tf.keras.Input(shape=(config.NUM_FEATURES,),
         name='Input'
    ))
    

    classifier_plus_model.add(keras.layers.Dense(
        units=config.FF_DIM,
        activation="relu",
        name='FF'
    ))

    
    classifier_plus_model.add(keras.layers.Dense(
        units=config.FF_LL_DIM,  
        use_bias=False,
        activation='softmax',
        name="Output"
    ))
  
   # compile
    classifier_plus_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = config.LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return classifier_plus_model
