import tensorflow as tf
from MultiHeadedAttention_IPU import *

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
  #def __init__(self,  d_model = 200, num_heads = 2, dff = 400, dropout = 0.0):
    super(DecoderLayer, self).__init__()

    self.multi_head_attention1 =  MultiHeadAttention_IPU(64, num_heads, causal = True)
    self.dropout_attention1 = tf.keras.layers.Dropout(dropout)
    self.add_attention1 = tf.keras.layers.Add()
    self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.multi_head_attention2 =  MultiHeadAttention_IPU(64, num_heads)
    self.dropout_attention2 = tf.keras.layers.Dropout(dropout)
    self.add_attention2 = tf.keras.layers.Add()
    self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    #print('inputs are: ', inputs)
    attention = self.multi_head_attention1([inputs[0],inputs[0],inputs[0]], mask = [mask[0],mask[0]])
    #print('after first attention: ', attention)
    attention = self.dropout_attention1(attention, training = training)
    #print('after second attention: ',attention)
    x = self.add_attention1([inputs[0] , attention])
    x = self.layer_norm_attention1(x)
    #print('after lauer norm! ', x)
    attention = self.multi_head_attention2([x, inputs[1],inputs[1]], mask = [mask[0],mask[1]])
    attention = self.dropout_attention2(attention, training = training)
    x = self.add_attention1([x , attention])
    x = self.layer_norm_attention1(x)
    #print('adter lauer norm 2: ', x)

    ## Feed Forward
    dense = self.dense1(x)
    #print('after dense 1: ', dense)
    dense = self.dense2(dense)
    #print('after dense 2: ', dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)

    return x
