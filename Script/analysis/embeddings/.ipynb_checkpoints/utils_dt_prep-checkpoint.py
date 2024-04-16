import pandas as pd
import numpy as np
import os
import glob
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
tf.keras.backend.set_floatx('float64')
import warnings
warnings.filterwarnings('ignore')

# set working directories
#########################
in_dir = 'C:/Users/cilin/Research/CA_hospitals/Input/final_data/all_combined/'
in_dir_codes = 'C:/Users/cilin/Research/CA_hospitals/Input/raw_data/health/'


# read data
###########
def read_data_bpe(keep_all_cols=False):
    df = pd.read_csv(in_dir + 'analysis_data_birth_pdd_edd.csv')
    df['data_source2'] = 'birth'
    
    # keep only cols of interest
    columns = [
    'rlnI_updated', 'bthdateI', 'bthyearI', 'cntyresI', 'cntyresI_name',
    'pm25I', 'wfeI', 'sexI', 'raceI', 
    'patcnty', 'admtdate', 'admtyear', 'admtmonth',
    'raceM', 'meduc', 'precare', 'visitsM_9mpp', 'visitsM_1ypp', 'visitsI_1yol',
    'bthresmb_name', 'prevsts',
    'diag00', 'diag01', 'diag02', 'diag03', 'diag04', 'data_source', 'data_source2'
    ]
    
    if keep_all_cols==True:
        return df
    else:
        return df[columns]

def read_data_pe():
    df = pd.read_csv(in_dir + 'analysis_data_pdd_edd.csv')
    df['data_source2'] = 'pdd_edd'
    # keep only cols of interest
    columns = [
    'rlnI_updated', 'bthdateI','bthyearI', 'cntyresI',
    'pm25I', 'wfeI', 'sexI', 'raceI', 
    'patcnty', 'admtdate', 'admtyear', 'admtmonth',
    'visitsM_9mpp', 'visitsM_1ypp', 'visitsI_1yol',
    'diag00', 'diag01', 'diag02', 'diag03', 'diag04', 'data_source', 'data_source2'
    ]
    return df[columns]

# drop observations
###################
def drop_observations(df, pat_min_length):
    ''''''
    # drop if admt month or year is missing (right now nothing is nan)
    df.dropna(subset=['admtmonth', 'admtyear'], inplace=True)

    # keep only patients with at least 3 hospital/ER visits
    temp = df.groupby('rlnI_updated',as_index=False).admtdate.count()
    temp = temp[(temp.admtdate.ge(pat_min_length))]
    df = df[df.rlnI_updated.isin(temp.rlnI_updated.unique())]
    
    # sort by admtdate
    df.sort_values(['rlnI_updated', 'admtdate'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('Drop observations: done')
    return df

   
# add features 
##############
def add_features(df, diag_per_visit, diag_len):
    ''''''
    # set rlnI_col
    rlnI_col = 'rlnI_updated'
    
    # create _2d or _3d diagnosis length
    dls = '_' + str(diag_len) + 'd' #diagnosis lengh string (e.g., '_3d')
    
    # decide how many diagnosis codes to consider from a visit
    if diag_per_visit==2:
        diagnosis = ['diag00', 'diag01']
        diagnosis_dl = ['diag00'+dls, 'diag01'+dls]
        
    if diag_per_visit==3:
        diagnosis = ['diag00', 'diag01', 'diag02']
        diagnosis_dl = ['diag00'+dls, 'diag01'+dls, 'diag02'+dls]
        
    if diag_per_visit==5:
        diagnosis = ['diag00', 'diag01', 'diag02', 'diag03', 'diag04']
        diagnosis_dl = ['diag00'+dls, 'diag01'+dls, 'diag02'+dls, 'diag03'+dls, 'diag04'+dls]
        
    # add age
    df['admtdate'] = pd.to_datetime(df.admtdate)
    df['bthdateI'] = pd.to_datetime(df.bthdateI)
    df['age'] = np.round((df.admtdate - df.bthdateI)/np.timedelta64(1, 'Y'), 1)
    df['age'] = df.age.astype(str)

    # add admtyear_patcnty
    df['admtyear_patcnty'] = df.admtyear.astype(str).str.split('.').str[0] + '_' +\
                             df.patcnty.astype(str).str.split('.').str[0] 
    
    # add bthyear_bthcnty
    df['bthyear_bthcnty'] = df.bthyearI.astype(str).str.split('.').str[0] + '_' +\
                         df.cntyresI.astype(str).str.split('.').str[0] 
    
    
    # add average pm25I (at the bthyear_bthcnty level) if pm25I is missing
    temp_df = df.groupby('bthyear_bthcnty', as_index=False).pm25I.mean()
    temp_df.rename(columns={'pm25I': 'pm25I_mean'}, inplace=True)
    df = df.merge(
        temp_df, 
        on='bthyear_bthcnty',
        how='left'
    )
    df['pm25I'] = np.where(df.pm25I.isna(), df.pm25I_mean, df.pm25I)
    
    
    # create 3 digit diagnosis codes (sub-chapter level) 
    for val in diagnosis:
        df[val] = df[val].astype(str)
        df[val+dls] = df[val].str[:diag_len]
    
    ## replace icd10 codes with icd9 codes ##
    # add start of icd10 codes
    df['start_ICD10'] = np.where((df.admtmonth.ge(9) & df.admtyear.ge(2015)), 1, 0)
    df['start_ICD10'] = np.where((df.admtyear.ge(2016)), 1, df.start_ICD10)

    # import icd10 to icd9 conversion
    codes = icd9_to_icd10(diag_len) 
    # replace for each diag code
    for val in diagnosis_dl:
        print(val)
        # merge codes from codes_dict for each key
        df = df.merge(
            codes,
            left_on=val.split('_')[0],
            right_on='ICD10'+dls,
            how='left'
        ) 
        
        # icd10 codes that exist before 2015Q4
        firstletter = (
            'F', 'G', 'J', 'M', 'T', 'O', 'I', 'N', 'D', 'A', 'K', 'S',
            'Q', 'Z', 'R', 'L', 'H', 'C', 'B', 'P'
        )
        
        # replace icd10codes if condition satisfied
        df[val] = np.where(df.start_ICD10.eq(1), df['ICD9'+dls], df[val])
        df[val] = np.where((df.start_ICD10.eq(0) & df[val].str.startswith(firstletter)), df['ICD9'+dls], df[val])
        df[val] = df[val].astype(str)
        
        df.drop(columns=['ICD10'+dls, 'ICD9'+dls], inplace=True)
        df[val] = np.where(df[val].eq('na'), 'nan', df[val])
 
    # creaate patient visit summary (diag, age, county)
    df = patient_visit(df, diag_per_visit, rlnI_col, diag_len)
    
    # sort by admtdate
    df.sort_values([rlnI_col, 'admtdate'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('Add observations: done')
    return df


# helpers
def icd9_to_icd10(diag_len):
    ''''''
    dls = '_' + str(diag_len) + 'd' #diagnosis lengh string (e.g., '_3d')
    codes = pd.read_csv(in_dir_codes + 'ICD_9_10_conversion.csv')
    #return codes
    # focus only on 2d codes
    codes['ICD10'+dls] = codes['TargetI9|Flags|I9Name'].str.split('|').str[0].str.replace('.', '').str[:diag_len]
    codes['ICD9'+dls] = codes['TargetI9|Flags|I9Name'].str.split('|').str[1].str.replace('.', '').str[:diag_len]
    codes = codes[['ICD10'+dls, 'ICD9'+dls]]
    # drop duplicates
    codes.drop_duplicates(['ICD10'+dls], inplace=True)
    return codes

def patient_visit(df, diag_per_visit, rlnI_col, diag_len):
    ''''''
    ## create patient diag per visit ##
    ###################################      
    dls = '_' + str(diag_len) + 'd' #diagnosis lengh string (e.g., '_3d')
    # add all diagnosis codes in a visit together #   
    if diag_per_visit==2:
        df['diag_list_visit'] = df['diag00'+dls] +  ' ' + df['diag01'+dls]
        
    if diag_per_visit==3:
        df['diag_list_visit'] = df['diag00'+dls] +  ' ' + \
                                df['diag01'+dls] +  ' ' + \
                                df['diag02'+dls]
        
    if diag_per_visit==5:
        df['diag_list_visit'] = df['diag00'+dls] +  ' ' +\
                                df['diag01'+dls] +  ' ' +\
                                df['diag02'+dls] +  ' ' +\
                                df['diag03'+dls] +  ' ' +\
                                df['diag04'+dls]
    # replace nans
    for val in ['na', 'nan', 'n']:
        df['diag_list_visit'] = df['diag_list_visit'].str.replace(val, '')
    # strip
    df['diag_list_visit'] = df['diag_list_visit'].str.strip()
    
    # add [SEP] to diag_list_visit to show the end of visit
    df['diag_list_visit'] = df['diag_list_visit'] + ' [SEP]'
    
    ## create patient age per visit ##
    ##################################
    # add age in a visit together 
    df['len_list_visits'] = df.diag_list_visit.str.split(' ').str.len()
    df['age_list_visit'] = df.len_list_visits * (' ' + df.age)
    
    ## create patient cnty per visit ##
    ##################################
    # transform cnty to string and remove .0
    for col in ['cntyresI', 'patcnty']:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.0', '')
    # add cnty in a visit together 
    df['len_list_visits'] = df.diag_list_visit.str.split(' ').str.len()
    df['cnty_list_visit'] = df.len_list_visits * (' ' + df.patcnty)
    return df


# add history
#############
def add_history(df, RD=False):
    ''''''
    # set rlnI_col
    rlnI_col = 'rlnI_updated' #rlnI_updated_pp
    
    if RD==True:
        rlnI_col = 'rlnI_updated_pp' 

    # create dictionary
    hist_dict = {}

    # add patient diag history
    df_diag_hist = patient_diag_hist(df, rlnI_col)
    hist_dict['diag'] = df_diag_hist
    del df_diag_hist
    print('Add patient diagnosis history: done')

    # add patient age history
    df_age_hist = patient_age_hist(df, rlnI_col)
    hist_dict['age'] = df_age_hist
    del df_age_hist
    print('Add patient age history: done')

    # add patient cnty history
    df_cnty_hist = patient_cnty_hist(df, rlnI_col)
    hist_dict['cnty'] = df_cnty_hist
    del df_cnty_hist
    print('Add patient cnty history: done')
    return hist_dict


# helpers    
def patient_diag_hist(df, rlnI_col):    
    ## create patient diag history ##
    #################################
    ## create patient diagnosis history
    pat_diag_hist = df.groupby(rlnI_col, as_index=False)['diag_list_visit'].apply(
        lambda x: ' '.join(x))
    pat_diag_hist.rename(columns={'diag_list_visit':"pat_diag_hist"}, inplace=True)
    
    # add [CLS] token at the beginning of the diag history
    pat_diag_hist['pat_diag_hist'] = '[CLS]' + ' ' + pat_diag_hist.pat_diag_hist
    pat_diag_hist.reset_index(drop=True, inplace=True)
    
    return pat_diag_hist

def patient_age_hist(df, rlnI_col):
    ''''''
    ## create patient age history ##
    ################################
    # patient age hist
    pat_age_hist = df.groupby(rlnI_col, as_index=False)['age_list_visit'].apply(
        lambda x: ''.join(x))
    pat_age_hist.rename(columns={'age_list_visit':"pat_age_hist"}, inplace=True)
    
    # patient min age
    pat_min_age = df.groupby(rlnI_col, as_index=False)['age'].min()
    pat_min_age.rename(columns={'age':"pat_min_age"}, inplace=True)

    # merge to df
    pat_age_hist = pat_age_hist.merge(pat_min_age, 
            on=rlnI_col,
            how='left')
    
    # add first age for [CLS] token at the beginning of the age history
    pat_age_hist['pat_age_hist'] = pat_age_hist.pat_min_age.astype(str) + pat_age_hist.pat_age_hist 
    pat_age_hist.drop(columns=['pat_min_age'], inplace=True)
    pat_age_hist.reset_index(drop=True, inplace=True)
    
    return pat_age_hist


def patient_cnty_hist(df, rlnI_col):
    ''''''
    ## create patient cnty history ##
    ################################
    # patient cnty hist
    pat_cnty_hist = df.groupby(rlnI_col, as_index=False)['cnty_list_visit'].apply(
        lambda x: ''.join(x))
    pat_cnty_hist.rename(columns={'cnty_list_visit':"pat_cnty_hist"}, inplace=True)

    # patient cnty at birth
    pat_cnty_first = df.groupby(rlnI_col, as_index=False)['patcnty'].first() 
    pat_cnty_first.rename(columns={'patcnty':"pat_cnty_first"}, inplace=True)

    # merge to df
    pat_cnty_hist = pat_cnty_hist.merge(pat_cnty_first, 
            on=rlnI_col,
            how='left')

    # add birth cnty for [CLS] token at the beginning of the cnty history
    pat_cnty_hist['pat_cnty_hist'] = pat_cnty_hist.pat_cnty_first.astype(str) + pat_cnty_hist.pat_cnty_hist 
    pat_cnty_hist.drop(columns=['pat_cnty_first'], inplace=True)
    pat_cnty_hist.reset_index(drop=True, inplace=True)
    
    return pat_cnty_hist
    

# diag processor
##################
def custom_standardization(input_data):
    #str_lowercase = tf.strings.lower(input_data)
    return input_data

# vectorize diag
##################
def get_vectorize_layer(texts,  max_seq, special_tokens=None):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=None, #can set vocab_size
        output_mode="int",
        standardize=None, #or custom_standardization function
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    if special_tokens:
        vocab = vectorize_layer.get_vocabulary()
        vocab = vocab[2 : len(vocab) - len(special_tokens)] + ["[MASK]"]
        vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer

# encode diag
#############
def encode(vectorize_layer, texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()

# save vectorize layer
######################
def save_vectorize_layer(vectorize_layer, key):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    
    model.add(vectorize_layer[key])
    model.save('./vectorizers/vect_layer_'+key, save_format=".h5")



# masked diag and labels
########################
def get_masked_input_and_labels(encoded_texts, mask_token_id, loc_CLS):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens (includes CLS and SEP)
    inp_mask[encoded_texts < 3] = False #(changes from True (masked) to False (unmasked))
    inp_mask[encoded_texts == loc_CLS] = False #(changes from True (masked) to False (unmasked)) #this for [CLS]
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens (leave the value of unmasked tokens as -1)
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] for 90% of the masked tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[inp_mask_2mask] = mask_token_id  # mask token is the last in the dict

    # Set the remaining 10% of the masked inputs to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        4, mask_token_id, inp_mask_2random.sum())

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0 # if masked input, then weight==1; if unmasked input then weight==0

    # y_labels would be same as encoded_texts i.e input tokens (this is the ground truth)
    y_labels = np.copy(encoded_texts)
    
    return encoded_texts_masked, y_labels, sample_weights
