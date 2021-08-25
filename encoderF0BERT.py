import tensorflow as tf
from utils import *
from EncoderLayer import *
from tensorflow.keras import layers


class F0BERTEncoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, num_layers=4, d_model=64, num_heads=2, dff=128, maximum_position_encoding=10000, dropout = 0.0):
  #def __init__(self, input_vocab_size, num_layers=4, d_model=64, num_heads=2, dff=128, maximum_position_encoding=10000, dropout = 0.5):
     
    super(F0BERTEncoder, self).__init__()

    self.d_model = d_model

    self.embeddingF0 = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    self.embeddingBERT = tf.keras.layers.Embedding(1, d_model)
    
    self.conv1 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")
    self.conv2 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")
    self.conv3 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")

    #self.conv4 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")
    #self.conv5 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")
    #self.conv6 = layers.Conv1D(filters = d_model, kernel_size= 3, padding="same", activation="relu")

    
    
    self.dense = layers.Dense(d_model)
    self.concat = layers.Concatenate(axis=-1)
    self.repeat = layers.RepeatVector(100)
    
    self.pos = positional_encoding(maximum_position_encoding, d_model)
    self.encoder_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, F0, BERT, mask=None, training=None):

    x = self.embeddingF0(F0)

    #BERT = self.repeat(BERT)
    BERT = self.embeddingBERT(BERT)
    x = self.concat([x, BERT])
    
    #channel 1
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    #channel 2, cannot apply con1D on BERT without adding first an embedding layer
    #BERT = self.conv4(BERT)
    #BERT = self.conv5(BERT)
    #BERT = self.conv6(BERT)

    #merge channels
    #x = self.concat([x, BERT])
    
    # positional encoding
    x *= (tf.math.sqrt(tf.cast(self.d_model, tf.float32)))
    x += (self.pos[: , :tf.shape(x)[1], :])
    x = self.dropout(x, training=training)
    
    #Encoder layer
    embedding_mask = self.embeddingBERT.compute_mask(F0)


    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, mask = embedding_mask)

    return x

  def compute_mask(self, inputs, mask=None):
    return self.embeddingBERT.compute_mask(inputs)

