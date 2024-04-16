import numpy as np
import pandas as pd
from numpy import argmax
from sklearn.metrics import average_precision_score as APS
from sklearn.metrics import roc_auc_score as ROC_AUC
from sklearn.metrics import roc_curve, auc
from pprint import pprint
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')

## set-up configuration
#######################
@dataclass
class Config:
    K = 1 # number of top predictions
config = Config()



# define helpers
def decode(id2token, tokens):
    return " ".join([id2token[t] for t in tokens if t != 0])

def convert_ids_to_tokens(id2token, ids):
    return id2token[ids]



# prediction for one patient
def one_patient_K_predictions(sample_tokens, y_sample, mlm_sample_prediction, id2token, mask_token_id):
    ''''''
    # find index of masked token 
    masked_index = np.where(sample_tokens['diag'][0] == mask_token_id)
    masked_index = masked_index[0]

    # find prediction for masked token
    mask_prediction = mlm_sample_prediction[0][masked_index]

    # find top predictions
    top_indices = mask_prediction[0].argsort()[-config.K :][::-1] # if more than two masked words, pull the first one

    # find top probabilities
    values = mask_prediction[0][top_indices]

    # print top predictions and ground truth
    print('Top 1 token predictions ')
    print('------------------------')
    for i in range(len(top_indices)):
        p = top_indices[i]
        v = values[i]
        tokens = np.copy(sample_tokens['diag'][0])
        tokens[masked_index[0]] = p
        result = {
            "input_text": decode(id2token, sample_tokens['diag'][0]),
            "prediction": decode(id2token, tokens),
            "probability": v,
            "predicted mask token": convert_ids_to_tokens(id2token, p),
            "ground truth": decode(id2token, y_sample[0])
        }
        pprint(result)



def all_patients_K_predictions_and_accuracy(sample_tokens, y_sample, mlm_sample_prediction, id2token, mask_token_id):
    ''''''
    # flatten ground truth diagnosis (no masking applied)
    y_sample_dflat = y_sample.flatten()

    # flatten sample diagnosis (maskign applied)
    sample_tokens_dflat = sample_tokens['diag'].flatten()
    #print('\nSample tokens - flatten:')
    #print('Shape ', sample_tokens_dflat.shape)
    #display(sample_tokens_dflat)


    # find index of masked token 
    masked_index = np.where(sample_tokens_dflat == mask_token_id)[0]
    #print("\nMasked tokens - indeces:")
    #print('Shape ', masked_index.shape)
    #display(masked_index)


    # flatten mlm predictions
    mlm_sample_prediction_dflat = mlm_sample_prediction.reshape(-1, mlm_sample_prediction.shape[-1])
    #display(mlm_sample_prediction_dflat)

    # find mlm predictions for masked token
    mask_prediction_dflat = mlm_sample_prediction_dflat[masked_index]
    #print("\nMasked tokens - prediction probabilities:")
    #print('Shape', mask_prediction_dflat.shape)
    #display(mask_prediction_dflat)

    # find index of top k predictions
    top_prediction_indices = mask_prediction_dflat.argsort()[:, -config.K]
    #print("\nMasked tokens - top predictions indeces:")
    #print('Shape', top_prediction_indices.shape)
    #display(top_prediction_indices)

    # find probabilities of top k predictions
    top_prediction_prob = []
    for idx in range(top_prediction_indices.shape[0]):
        top_prediction_prob.append(mask_prediction_dflat[idx, top_prediction_indices[idx]])
        
        
    # ground truth of masked token
    ground_truth_masked_token = []

    temp = y_sample_dflat[masked_index]

    for idx in range(temp.shape[0]):
        ground_truth_masked_token.append(convert_ids_to_tokens(id2token, temp[idx]))


    # precitions of masked token
    prediction_masked_token = []
    for idx in range(top_prediction_indices.shape[0]):
        prediction_masked_token.append(convert_ids_to_tokens(id2token, top_prediction_indices[idx]))

    # compute prediction error/accuracy
    error = (np.array(ground_truth_masked_token) != np.array(prediction_masked_token)).sum()/len(ground_truth_masked_token)
    accuracy = 1-error
    return accuracy




def all_patients_predictions_and_apc(vectorize_layer, sample_tokens, y_sample, mlm_sample_prediction, id2token, sample_sample_weights):
    ''''''
    # define input string
    data = [convert_ids_to_tokens(id2token, val) for val in y_sample[sample_sample_weights == 1]]
    #print(data)

    # define universe of possible input values
    alphabet = vectorize_layer['diag'].get_vocabulary()

    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))


    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    #print(integer_encoded)
    
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
    #print(onehot_encoded)
            
    y_sample_onehot = np.array(onehot_encoded)
    #print('Shape of y_sample_onehot ', y_sample_onehot.shape)


    # find predictions for masked tokens
    y_sample_pred = mlm_sample_prediction[:][sample_sample_weights == 1]
    #print('Shape of y_sample_pred ', y_sample_pred.shape)

    
    aps_samples = []
    fpr_micro= []
    tpr_micro = []
    area_micro = []

    # compute average precision score
    aps_samples = APS(
        y_sample_onehot,
        y_sample_pred,
        average='samples'
    )
    print('APS:', aps_samples)
    
    # ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(
        y_sample_onehot.ravel(),
        y_sample_pred.ravel()
    )
    area_micro = auc(fpr_micro, tpr_micro)
    print('AUC:', area_micro)
    
    
    # add ROC area and APS to a df
    temp_df = pd.DataFrame(
        {'metric': ['APS', 'AUC'],
         'value': [aps_samples, area_micro]
        }
    )
    return temp_df
    
    
