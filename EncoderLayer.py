import tensorflow as tf
from MultiHeadedAttention import *

class EncoderLayer(tf.keras.layers.Layer):
  #def __init__(self,  d_model = 200, num_heads = 2, dff = 400, dropout = 0.0):
  def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
    super(EncoderLayer, self).__init__()

    self.multi_head_attention =  MultiHeadAttention(d_model, num_heads)
    self.dropout_attention = tf.keras.layers.Dropout(dropout)
    self.add_attention = tf.keras.layers.Add()
    self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    #print('mask: ', mask)
    #print('    inputs: ', inputs)
    #print('    mask: ', mask)
    attention = self.multi_head_attention([inputs,inputs,inputs], mask = [mask,mask])
    #print('    after first attention: ', attention)
    attention = self.dropout_attention(attention, training = training)
    #print('    after second attention: ', attention)
    x = self.add_attention([inputs , attention])
    x = self.layer_norm_attention(x)
    #print('    after norm attention x: ', x)
    # x = inputs

    ## Feed Forward
    dense = self.dense1(x)
    #print('    after first dense: ', dense)
    dense = self.dense2(dense)
    #print('    after sencond dense: ', dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)
    #print('HI I AM INSIDE ENCODER LAYER/ ', x)
    return x

