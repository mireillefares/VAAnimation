import tensorflow as tf
from utils import *
from DecoderLayer2 import *
#from DecoderLayer import *

class BERTDecoder(tf.keras.layers.Layer):
  #def __init__(self, num_layers = 1, d_model = 64, num_heads = 2, dff = 128, maximum_position_encoding = 10000, dropout = 0.0):
  def __init__(self, num_layers = 1, d_model = 64, num_heads = 2, dff = 128, maximum_position_encoding = 10000, dropout = 0.0):
    super(BERTDecoder, self).__init__()

    self.d_model = d_model

    #self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
    self.pos = positional_encoding(maximum_position_encoding, d_model)

    self.decoder_layers = [DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)  for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask=None, training=None):
    #print('   ')
    #print('   decoding: ')
    x = (inputs[0])
    # positional encoding
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos[: , :tf.shape(x)[1], :]
    #x = self.dropout(x, training=training)
    #print('x after positional encoding: ', x)
    #Decoder layer
    #embedding_mask = self.embedding.compute_mask(inputs[0])
    #print('embedding mask is : ', embedding_mask)
    #print('mask is : ', mask)
    for decoder_layer in self.decoder_layers:
      x = decoder_layer([x,inputs[1]], mask = [None, None])
      #print(' inside loop of decoding, after decoding layer:  ', x)
      

    return x

  # Comment this out if you want to use the masked_loss()
  def compute_mask(self, inputs, mask=None):
    #return self.embedding.compute_mask(inputs[0])
    return None
