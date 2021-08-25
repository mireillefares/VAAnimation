import tensorflow as tf
from utils import *
from EncoderLayer import *
from tensorflow.keras import layers


class BertEncoder(tf.keras.layers.Layer):
  #zabta 
  #def __init__(self, num_layers=4, d_model= 768, num_heads= 6, dff=128, maximum_position_encoding=10000, dropout = 0.0):
  #def __init__(self, num_layers=1, d_model= 768, num_heads= 12, dff=128, maximum_position_encoding=10000, dropout = 0.5):   
  def __init__(self, num_layers=1, d_model= 768, num_heads= 12, dff=128, maximum_position_encoding=10000, dropout = 0.0):
    super(BertEncoder, self).__init__()

    self.embedding = tf.keras.layers.Embedding(1, d_model)
    self.d_model = d_model

    self.dense = layers.Dense(d_model)
    self.concat = layers.Concatenate(axis=-1)
    self.repeat = layers.RepeatVector(100)
    
    self.pos = positional_encoding(maximum_position_encoding, d_model)

    self.encoder_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, BERT, mask=None, training=None):

    x = BERT

    
    # positional encoding
    x *= (tf.math.sqrt(tf.cast(self.d_model, tf.float32)))
    x += (self.pos[: , :tf.shape(x)[1], :])
    x = self.dropout(x, training=training)
    

    # Encoder layer
    #embedding_mask = self.embedding.compute_mask(BERT)

    for encoder_layer in self.encoder_layers:
      #x = encoder_layer(x, mask = embedding_mask)
      x = encoder_layer(x, mask = None)
    return x

  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs)


  def compute_output_shape(self, input_shape):
    shape = [None , 100, 768]
    return shape

            
