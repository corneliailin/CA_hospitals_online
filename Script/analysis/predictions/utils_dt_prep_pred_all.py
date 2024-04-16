import pandas as pd
import numpy as np
import os
import glob
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
tf.keras.backend.set_floatx('float64')
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../')
from embeddings import utils_dt_prep
from  embeddings import utils_MLM
  


# input_output pairs
#####################
def input_output_pairs(df, pat_min_length):
    ''''''
    # compute total visits per patient
    temp = df.groupby('rlnI_updated',as_index=False).admtdate.count()
    temp.rename(columns={'admtdate':'total_visits'}, inplace=True)

    # choose a visit number at random to split the data into input and output
    random_t = []
    for val in temp.total_visits:
        # make sure you have                                             
        interval = np.arange(pat_min_length-1, val) # leave at least pat_min_length-1 visits in patient input 
        nr = int(np.random.choice(interval))
        random_t.append(nr)
    temp['random_t'] = random_t

    # merge it back to df
    df = df.merge(
        temp, 
        on='rlnI_updated',
        how='left'
    )
    df.reset_index(drop=True, inplace=True)
    
    # add index at the grp level
    df['_1s'] = 1
    df['index_grp'] = df.groupby('rlnI_updated')._1s.cumsum()
    
    # create input-output df pairs
    df_in= df[df.index_grp.le(df.random_t)]
    df_out = df[df.index_grp.eq(df.random_t+1)]
    df_in.reset_index(drop=True, inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    return df_in, df_out

