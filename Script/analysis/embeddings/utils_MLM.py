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
config = Config()

## text generator example ##
############################
class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, vectorize_layer, sample_tokens, id2token, token2id, mask_token_id, top_k=5):
        self.sample_tokens = sample_tokens['diag'][0]
        self.id2token = dict(enumerate(vectorize_layer['diag'].get_vocabulary()))
        self.token2id = {y: x for x, y in self.id2token.items()}
        self.mask_token_id = mask_token_id
        self.k = top_k

    def decode(self, tokens):
        return " ".join([self.id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[0][1]
        mask_prediction = prediction[0][masked_index]
        top_indices = mask_prediction[0].argsort()[-self.k :][::-1] # if more than two masked words, pull the first one
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens)
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(self.sample_tokens),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)


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


# loss function
###############
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")

## custom training function ##
##############################
class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):

        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            print('More than 3 inputs')
        
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]
   


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
def create_masked_language_bert_model(vectorize_layer, sample_weights, keys):
    '''
    '''
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

    ## ENCODER ##
    #############
    # The encoder receives the embeddings
    encoder_output = embeddings
    for i in range(config.NUM_LAYERS): # only 1 layer
        
        ## Multi-Head self-attention ##
        ###############################
        query =  encoder_output
        key = encoder_output
        value = encoder_output

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

        encoder_output = sequence_output

    # includes the multi-head attention, and fully connected layer
    mlm_output = layers.Dense(
        len(vectorize_layer['diag'].get_vocabulary()),
        name="mlm_cls",
        activation="softmax")(encoder_output)
        
    mlm_model = MaskedLanguageModel(
        inputs=input_list,
        outputs = mlm_output,
        name="masked_bert_model"
    )
    
    ## Compile ##
    #############
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    
    mlm_model.compile(optimizer=optimizer
                      #loss_weights=sample_weights,
                      #metrics=['accuracy']
    )

    return mlm_model
