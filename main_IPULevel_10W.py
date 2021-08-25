#https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83
#https://colab.research.google.com/drive/1CBe2VlogbyXzmIyRQGH5xzuvLwGrvjcf?usp=sharing#scrollTo=VHU3F9_VE5rp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  BatchNormalization, Flatten, Activation, dot, Multiply, LeakyReLU, concatenate, Lambda, Layer, Input, Dense, LSTM, TimeDistributed, Concatenate, RepeatVector, Bidirectional, Masking, Conv1D, Dropout, Add, MaxPool1D, MaxPooling1D,  GlobalMaxPool1D, add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from numpy import array, mean, argmax, array_equal
import numpy as np
from utils import *
import pandas as pd
import pickle 
import sys
import multiprocessing
from multiprocessing import Pool
from tensorflow.keras.callbacks import EarlyStopping
import manage_gpus as gpl
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import time, re, os, io
from itertools import chain
import random
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error
from encoderBert import *
from encoderF0 import *
from encoderF0BERT import *
from decoder import *
from BERTdecoder import *
from attention import AttentionLayer
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from keras_self_attention import SeqSelfAttention
from AttentionWithContext import AttentionWithContext   
from BERTdecoder import *
from pandas import DataFrame
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler
import itertools
from MultiHeadedAttention import *

K.clear_session()
np.set_printoptions(threshold=sys.maxsize)




def format_data(DFAllData_F0, DFAllData_AU01, DFAllData_AU02, DFAllData_AU04, DFAllData_AU05, DFAllData_AU06, DFAllData_AU07, DFAllData_Rx, DFAllData_Ry, DFAllData_Rz):

    DFAllData_F0 = DFAllData_F0.reset_index(level=[0, 1], drop = True)
    DFAllData_AU01 = DFAllData_AU01.reset_index(level=[0, 1], drop = True)
    DFAllData_AU02 = DFAllData_AU02.reset_index (level=[0, 1], drop = True)
    DFAllData_AU04 = DFAllData_AU04.reset_index(level=[0, 1], drop = True)
    DFAllData_AU05 = DFAllData_AU05.reset_index (level=[0, 1], drop = True)
    DFAllData_AU06 = DFAllData_AU06.reset_index(level=[0, 1], drop = True)
    DFAllData_AU07 = DFAllData_AU07.reset_index (level=[0, 1], drop = True)
    DFAllData_Rx = DFAllData_Rx.reset_index (level=[0, 1], drop = True)
    DFAllData_Rz = DFAllData_Rz.reset_index (level=[0, 1], drop = True)
    DFAllData_Ry = DFAllData_Ry.reset_index (level=[0, 1], drop = True)

    
    DFAllData_F0.insert(0, "IPU#", IPUNumber)
    DFAllData_AU01.insert(0, "IPU#", IPUNumber)
    DFAllData_AU02.insert(0, "IPU#", IPUNumber)
    DFAllData_AU04.insert(0, "IPU#", IPUNumber)
    DFAllData_AU05.insert(0, "IPU#", IPUNumber)
    DFAllData_AU06.insert(0, "IPU#", IPUNumber)
    DFAllData_AU07.insert(0, "IPU#", IPUNumber)
    DFAllData_Rx.insert(0, "IPU#", IPUNumber)
    DFAllData_Rz.insert(0, "IPU#", IPUNumber)
    DFAllData_Ry.insert(0, "IPU#", IPUNumber)     
    
    DFAllData_F0index  = DFAllData_F0.index
    DFAllData_AU01index= DFAllData_AU01.index
    DFAllData_AU02index= DFAllData_AU02.index
    DFAllData_AU04index= DFAllData_AU04.index
    DFAllData_AU05index= DFAllData_AU05.index
    DFAllData_AU06index= DFAllData_AU06.index
    DFAllData_AU07index= DFAllData_AU07.index
    DFAllData_Rxindex= DFAllData_Rx.index
    DFAllData_Rzindex= DFAllData_Rz.index
    DFAllData_Ryindex= DFAllData_Ry.index

    DFAllData_F0.reset_index()
    DFAllData_AU01.reset_index()
    DFAllData_AU02.reset_index()
    DFAllData_AU04.reset_index()
    DFAllData_AU05.reset_index()
    DFAllData_AU06.reset_index()
    DFAllData_AU07.reset_index()
    DFAllData_Rx.reset_index()
    DFAllData_Rz.reset_index()
    DFAllData_Ry.reset_index()

    DFAllData_F0.set_index(["IPU#", DFAllData_F0index], inplace=True)
    DFAllData_AU01.set_index(["IPU#", DFAllData_AU01index], inplace=True)
    DFAllData_AU02.set_index(["IPU#", DFAllData_AU02index], inplace=True)
    DFAllData_AU04.set_index(["IPU#", DFAllData_AU04index], inplace=True)
    DFAllData_AU05.set_index(["IPU#", DFAllData_AU05index], inplace=True)
    DFAllData_AU06.set_index(["IPU#", DFAllData_AU06index], inplace=True)
    DFAllData_AU07.set_index(["IPU#", DFAllData_AU07index], inplace=True)
    DFAllData_Rx.set_index(["IPU#", DFAllData_Rxindex], inplace=True)
    DFAllData_Rz.set_index(["IPU#", DFAllData_Rzindex], inplace=True)
    DFAllData_Ry.set_index(["IPU#", DFAllData_Ryindex], inplace=True)
                        
    
    #create padded input target sequence
    SOS = np.repeat(8, DFAllData_AU01.shape[0])
    AU01_in = DFAllData_AU01.copy()
    AU02_in = DFAllData_AU02.copy()
    AU04_in = DFAllData_AU04.copy()
    AU05_in = DFAllData_AU05.copy()
    AU06_in = DFAllData_AU06.copy()
    AU07_in = DFAllData_AU07.copy()
    Rx_ = DFAllData_Rx.copy()
    Rz_ = DFAllData_Rz.copy()
    Ry_ = DFAllData_Ry.copy()
                            
    AU01_in.insert(0, "SOS", SOS)
    AU02_in.insert(0, "SOS", SOS)
    AU04_in.insert(0, "SOS", SOS)
    AU05_in.insert(0, "SOS", SOS)
    AU06_in.insert(0, "SOS", SOS)
    AU07_in.insert(0, "SOS", SOS)
    Rx_.insert(0, "SOS", SOS)
    Ry_.insert(0, "SOS", SOS)
    Rz_.insert(0, "SOS", SOS)
    AU01_in = AU01_in.to_numpy()
    AU02_in = AU02_in.to_numpy()
    AU04_in = AU04_in.to_numpy()
    AU05_in = AU05_in.to_numpy()
    AU06_in = AU06_in.to_numpy()
    AU07_in = AU07_in.to_numpy()
    Rx_ = Rx_.to_numpy()
    Rz_ = Rz_.to_numpy()
    Ry_ = Ry_.to_numpy()
    AU01_in = formatDF_out(AU01_in)
    AU02_in = formatDF_out(AU02_in)
    AU04_in = formatDF_out(AU04_in)
    AU05_in = formatDF_out(AU05_in)
    AU06_in = formatDF_out(AU06_in)
    AU07_in = formatDF_out(AU07_in)
    Rx_ = formatDF_out(Rx_)
    Rz_ = formatDF_out(Rz_)
    Ry_ = formatDF_out(Ry_)
    
    AU01_in_np = (AU01_in)
    AU02_in_np = (AU02_in)
    AU04_in_np = (AU04_in)
    AU05_in_np = (AU05_in)
    AU06_in_np = (AU06_in)
    AU07_in_np = (AU07_in)
    Rx_np = Rx_
    Rz_np = Rz_
    Ry_np = Ry_
    
    AU01_in = pd.DataFrame(index = DFAllData_AU01.index, data=AU01_in_np)
    AU02_in = pd.DataFrame(index = DFAllData_AU02.index, data=AU02_in_np)
    AU04_in = pd.DataFrame(index = DFAllData_AU04.index, data=AU04_in_np)
    AU05_in = pd.DataFrame(index = DFAllData_AU05.index, data=AU05_in_np)
    AU06_in = pd.DataFrame(index = DFAllData_AU06.index, data=AU06_in_np)
    AU07_in = pd.DataFrame(index = DFAllData_AU07.index, data=AU07_in_np)
    Rx_ = pd.DataFrame(index = DFAllData_Rx.index, data=Rx_np)
    Rz_ = pd.DataFrame(index = DFAllData_Rz.index, data=Rz_np)
    Ry_ = pd.DataFrame(index = DFAllData_Ry.index, data=Ry_np)

    return DFAllData_F0, AU01_in, AU02_in, AU04_in, AU05_in, AU06_in, AU07_in, Rx_, Rz_, Ry_


def get_comp_device():
    board_ids = gpl.board_ids()
    if board_ids is None:
        return "/CPU:0"
    else:
        gpu_id_locked = gpl.obtain_lock_id(id=-1) # id = -1 locks an arbitrary free gpu
        if gpu_id_locked < 0:
            raise RuntimeError("no lock obtained")
        return "/GPU:0"

def encode(SPK, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11):
    return  SPK, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11
        

def tf_encode(spk, x, y, z1, z2,z4, z5, z6, z7, z8, z10, z11):
    y = tf.cast(y, tf.float32)
    result_spk, result_x, result_y, result_z1, result_z2, result_z4, result_z5, result_z6, result_z7,  result_z8, result_z10, result_z11 = tf.py_function(encode, [spk, x, y, z1, z2,z4, z5, z6, z7, z8, z10, z11], [tf.int64, tf.int64, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64,tf.int64,tf.int64, tf.int64, tf.int64, tf.int64])

    result_spk.set_shape([None, None])
    result_x.set_shape([None, None])
    result_y.set_shape([None, None])
    result_z1.set_shape([None])
    result_z2.set_shape([None])
    result_z4.set_shape([None])
    result_z5.set_shape([None])
    result_z6.set_shape([None])
    result_z7.set_shape([None])
    result_z8.set_shape([None])
    result_z10.set_shape([None])
    result_z11.set_shape([None])
    
    return result_spk, result_x, result_y, result_z1, result_z2, result_z4, result_z5, result_z6, result_z7, result_z8, result_z10, result_z11

    
def split_train_validate_test (F0, AU1_in, AU2_in, AU4_in, AU5_in, AU6_in, AU7_in, Rx, Ry, Rz, bert, IPUs_filtered, Speakers_filtered):
    LenBERT = bert.shape[0]
    LenF0 = F0.shape[0]
    LenAU = AU1_in.shape[0]
    
    #take out some successive IPUs for testing
    Successive1_IPUs = IPUs_filtered[:10]
    Successive1_F0 = F0[:10]
    Successive1_AU1 = AU1_in[:10]
    Successive1_AU2 = AU2_in[:10]
    Successive1_AU4 = AU4_in[:10]
    Successive1_AU5 = AU5_in[:10]
    Successive1_AU6 = AU6_in[:10]
    Successive1_AU7 = AU7_in[:10]
    Successive1_Rx = Rx[:10]
    Successive1_Rz = Rz[:10]
    Successive1_Ry = Ry[:10]
    Successive1_Speakers = Speakers_filtered[:10, :, :]
    Successive1_bert = bert[:10]

    Successive2_IPUs = IPUs_filtered[int(0.9*LenAU):]
    Successive2_F0 = F0[int(0.9*LenF0): ]
    Successive2_AU1 = AU1_in[int(0.9*LenAU): ]
    Successive2_AU2 = AU2_in[int(0.9*LenAU): ]
    Successive2_AU4 = AU4_in[int(0.9*LenAU): ]
    Successive2_AU5 = AU5_in[int(0.9*LenAU): ]
    Successive2_AU6 = AU6_in[int(0.9*LenAU): ]
    Successive2_AU7 = AU7_in[int(0.9*LenAU): ]
    Successive2_Rx = Rx[int(0.9*LenAU): ]
    Successive2_Ry = Ry[int(0.9*LenAU): ]
    Successive2_Rz = Rz[int(0.9*LenAU): ]
    Successive2_Speakers = Speakers_filtered[int(0.9*LenAU): , :, :]
    Successive2_bert = bert[int(0.9*LenBERT): ]
        
    IPUs_filtered = IPUs_filtered[10:]
    F0 = F0[10:]
    AU1_in = AU1_in[10:]
    AU2_in = AU2_in[10:]
    AU4_in = AU4_in[10:]
    AU5_in = AU5_in[10:]
    AU6_in = AU6_in[10:]
    AU7_in = AU7_in[10:]
    Rx = Rx [10:]
    Rz = Rz [10:]
    Ry = Ry [10:]
    Speakers_filtered = Speakers_filtered[10:, :, :]
    bert= bert[10:]
                                                    
    Successive1_F0 = np.array(Successive1_F0)
    Successive1_AU1 = np.array(Successive1_AU1)
    Successive1_AU2 = np.array(Successive1_AU2)
    Successive1_AU4 = np.array(Successive1_AU4)
    Successive1_AU5 = np.array(Successive1_AU5)
    Successive1_AU6 = np.array(Successive1_AU6)
    Successive1_AU7 = np.array(Successive1_AU7)
    Successive1_Rx = np.array(Successive1_Rx)
    Successive1_Ry = np.array(Successive1_Ry)
    Successive1_Rz = np.array(Successive1_Rz)
    Successive1_Speakers = np.array(Successive1_Speakers)
    Successive1_bert = np.array(Successive1_bert)
    Successive1_IPUs = np.array(Successive1_IPUs)

    
    Successive2_AU1 = np.array(Successive2_AU1)
    Successive2_AU2 = np.array(Successive2_AU2)
    Successive2_AU4 = np.array(Successive2_AU4)
    Successive2_AU5 = np.array(Successive2_AU5)
    Successive2_AU6 = np.array(Successive2_AU6)
    Successive2_AU7 = np.array(Successive2_AU7)
    Successive2_Rx = np.array(Successive2_Rx)
    Successive2_Ry = np.array(Successive2_Ry)
    Successive2_Rz = np.array(Successive2_Rz)
    Successive2_Speakers = np.array(Successive2_Speakers)
    Successive2_bert = np.array(Successive2_bert)
    Successive2_IPUs = np.array(Successive2_IPUs)
    
    
    #shuffle IPUs
    shuffled = list(zip(IPUs_filtered, Speakers_filtered, F0, bert, AU1_in, AU2_in, AU4_in, AU5_in, AU6_in, AU7_in, Rx, Ry, Rz))
    random.shuffle(shuffled)
    IPUs_filtered, Speakers_filtered, F0, bert, AU1_in, AU2_in, AU4_in, AU5_in, AU6_in, AU7_in, Rx, Ry, Rz = zip(*shuffled)

    
    LenBERT = np.array(bert).shape[0]
    LenF0 = np.array(F0).shape[0]
    LenAU = np.array(AU1_in).shape[0]
            
    
    F0train = F0[:int(0.9*LenF0)]
    AU1_train = AU1_in[:int(0.9*LenAU)]
    AU2_train = AU2_in[:int(0.9*LenAU)]
    AU4_train = AU4_in[:int(0.9*LenAU)]
    AU5_train = AU5_in[:int(0.9*LenAU)]
    AU6_train = AU6_in[:int(0.9*LenAU)]
    AU7_train = AU7_in[:int(0.9*LenAU)]
    Rx_train = Rx[:int(0.9*LenAU)]
    Ry_train = Ry[:int(0.9*LenAU)]
    Rz_train = Rz[:int(0.9*LenAU)]

    Speakers_train = Speakers_filtered[0:int(0.9*LenAU)]
    Bert_train = bert[:int(0.9*LenBERT)]

    F0validate = F0[int(0.9*LenF0):]
    Bert_validate = bert[int(0.9*LenBERT):]
    AU1_validate = AU1_in[int(0.9*LenAU):]
    AU2_validate = AU2_in[int(0.9*LenAU):]
    AU4_validate = AU4_in[int(0.9*LenAU):]
    AU5_validate = AU5_in[int(0.9*LenAU):]
    AU6_validate = AU6_in[int(0.9*LenAU):]
    AU7_validate = AU7_in[int(0.9*LenAU):]
    Rx_validate = Rx[int(0.9*LenAU):]
    Rz_validate = Rz[int(0.9*LenAU):]
    Ry_validate = Ry[int(0.9*LenAU):]
    Speakers_validate = Speakers_filtered[int(0.9*LenAU):]

    
    AU1_train = np.array(AU1_train)
    AU2_train = np.array(AU2_train)
    AU4_train = np.array(AU4_train)
    AU5_train = np.array(AU5_train)
    AU6_train = np.array(AU6_train)
    AU7_train = np.array(AU7_train)
    Rx_train = np.array(Rx_train)
    Rz_train = np.array(Rz_train)
    Ry_train = np.array(Ry_train)
    Speakers_train = np.array(Speakers_train)
    
    F0train = np.array(F0train)
    F0validate = np.array(F0validate)
    AU1_validate = np.array(AU1_validate)
    AU2_validate = np.array(AU2_validate)
    AU4_validate = np.array(AU4_validate)
    AU5_validate = np.array(AU5_validate)
    AU6_validate = np.array(AU6_validate)
    AU7_validate = np.array(AU7_validate)
    Rx_validate = np.array(Rx_validate)
    Rz_validate = np.array(Rz_validate)
    Ry_validate = np.array(Ry_validate)
    Speakers_validate = np.array(Speakers_validate)
    
    Bert_train = np.array(Bert_train)
    Bert_validate = np.array(Bert_validate)

                                                    
    return AU1_train, AU2_train, AU4_train, AU5_train, AU6_train, AU7_train, F0train, F0validate, AU1_validate,AU2_validate, AU4_validate, AU5_validate, AU6_validate, AU7_validate, Bert_train, Bert_validate, Rx_train, Ry_train, Rz_train, Rx_validate, Ry_validate, Rz_validate, Speakers_train, Speakers_validate, Successive1_F0, Successive2_F0, Successive1_AU1, Successive1_AU2, Successive1_AU4, Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive2_AU1, Successive2_AU2, Successive2_AU4, Successive2_AU5, Successive2_AU6, Successive2_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, Successive2_Rx, Successive2_Ry, Successive2_Rz, Successive1_Speakers, Successive2_Speakers, Successive2_bert, Successive1_bert, Successive1_IPUs, Successive2_IPUs



def format_IPULevel(IPUs, DFAllData_F0, AU01_in, AU02_in, AU04_in, AU05_in, AU06_in, AU07_in, Rx_DF, Ry_DF, Rz_DF, BERT_new):

    #bert
    dict_bert = {}
    keys = list(range(IPUs.shape[0]))
    bert = []
    BERT = []
    F0 = []
    AU1_in = []
    AU2_in = []
    AU4_in = []
    AU5_in = []
    AU6_in = []
    AU7_in = []
    RX_DF = []
    RZ_DF = []
    RY_DF = []

    #convert IPU DataFrame to numpy array
    IPUs_array = IPUs.to_numpy()

    #speakers
    indexes = IPUs.index.values
    speakers = []
    for i in indexes:
        speakers.append(i[0])

    IPUs_filtered = []
    Speakers_Filtered = []
        
    #loop over IPUs    
    for i in keys:        
        if i == 0:
            if(DFAllData_F0.loc[i].shape[0] == (len(BERT_new[i][1:-1]))):
                IPUs_filtered.append(IPUs_array[i])
                Speakers_Filtered.append(speakers[i])
                NbWords = (DFAllData_F0.loc[i].shape[0])
                if NbWords < 10:
                    Nb_to_add = 10 - DFAllData_F0.loc[i].shape[0]
                    rowF0 = DFAllData_F0.iloc[0]
                    rowAU = AU01_in.iloc[0]
                    rowF0 = rowF0.replace(rowF0.values, -1.0)
                    rowAU = rowAU.replace(rowAU.values, 7.0)
                    rowBERT = np.repeat(0, 768)
                    F0_toappend = DFAllData_F0.loc[i]
                    AU1_toappend = AU01_in.loc[i]
                    AU2_toappend = AU02_in.loc[i]
                    AU4_toappend = AU04_in.loc[i]
                    AU5_toappend = AU05_in.loc[i]
                    AU6_toappend = AU06_in.loc[i]
                    AU7_toappend = AU07_in.loc[i]
                    Rx_toappend = Rx_DF.loc[i]
                    Rz_toappend = Rz_DF.loc[i]
                    Ry_toappend = Ry_DF.loc[i]

                    arr = []
                    for j in BERT_new[i][1:-1]:
                        bert.append(j[1])
                        arr.append(j[1])
                    
                    for i in range(0, Nb_to_add):
                        F0_toappend = F0_toappend.append(rowF0)
                        AU1_toappend = AU1_toappend.append(rowAU)
                        AU2_toappend = AU2_toappend.append(rowAU)
                        AU4_toappend = AU4_toappend.append(rowAU)
                        AU5_toappend = AU5_toappend.append(rowAU)
                        AU6_toappend = AU6_toappend.append(rowAU)
                        AU7_toappend = AU7_toappend.append(rowAU)
                        Rx_toappend = Rx_toappend.append(rowAU)
                        Rz_toappend = Rz_toappend.append(rowAU)
                        Ry_toappend = Ry_toappend.append(rowAU)                    
                        arr.append(rowBERT)
                        bert.append(rowBERT)

                    F0.append(F0_toappend.to_numpy())
                    AU1_in.append(AU1_toappend.to_numpy())
                    AU2_in.append(AU2_toappend.to_numpy())
                    AU4_in.append(AU4_toappend.to_numpy())
                    AU5_in.append(AU5_toappend.to_numpy())
                    AU6_in.append(AU6_toappend.to_numpy())
                    AU7_in.append(AU7_toappend.to_numpy())
                    RX_DF.append(Rx_toappend.to_numpy())
                    RZ_DF.append(Rz_toappend.to_numpy())
                    RY_DF.append(Ry_toappend.to_numpy())
                    BERT.append(np.array(arr))
                
                if NbWords >= 10:
                    F0_toappend = DFAllData_F0.loc[i].head(10)
                    AU1_toappend = AU01_in.loc[i].head(10)
                    AU2_toappend = AU02_in.loc[i].head(10)
                    AU4_toappend = AU04_in.loc[i].head(10)
                    AU5_toappend = AU05_in.loc[i].head(10)
                    AU6_toappend = AU06_in.loc[i].head(10)
                    AU7_toappend = AU07_in.loc[i].head(10)
                    Rx_toappend = Rx_DF.loc[i].head(10)
                    Rz_toappend = Rz_DF.loc[i].head(10)
                    Ry_toappend = Ry_DF.loc[i].head(10)
                
                    arr = []
                    for j in BERT_new[i][1:11]:
                        bert.append(j[1])
                        arr.append(j[1])
                    dict_bert[i] = arr
                    BERT.append(np.array(arr))
                    F0.append(F0_toappend.to_numpy())
                    AU1_in.append(AU1_toappend.to_numpy())
                    AU2_in.append(AU2_toappend.to_numpy())
                    AU4_in.append(AU4_toappend.to_numpy())
                    AU5_in.append(AU5_toappend.to_numpy())
                    AU6_in.append(AU6_toappend.to_numpy())
                    AU7_in.append(AU7_toappend.to_numpy())
                    RX_DF.append(Rx_toappend.to_numpy())
                    RZ_DF.append(Rz_toappend.to_numpy())
                    RY_DF.append(Ry_toappend.to_numpy())
                
        if i>0 and (i in DFAllData_F0.index):
            if(DFAllData_F0.loc[i].shape[0] == (len(BERT_new[i][1:-1]))):
                IPUs_filtered.append(IPUs_array[i])
                Speakers_Filtered.append(speakers[i])
                NbWords = (DFAllData_F0.loc[i].shape[0])
                if NbWords < 10:
                    Nb_to_add = 10 - DFAllData_F0.loc[i].shape[0]
                    rowF0 = DFAllData_F0.iloc[0]
                    rowAU = AU01_in.iloc[0]
                    rowF0 = rowF0.replace(rowF0.values, -1.0)
                    rowAU = rowAU.replace(rowAU.values, 7.0)
                    rowBERT = np.repeat(0, 768)
                    F0_toappend = DFAllData_F0.loc[i]
                    AU1_toappend = AU01_in.loc[i]
                    AU2_toappend = AU02_in.loc[i]
                    AU4_toappend = AU04_in.loc[i]
                    AU5_toappend = AU05_in.loc[i]
                    AU6_toappend = AU06_in.loc[i]
                    AU7_toappend = AU07_in.loc[i]
                    Rx_toappend = Rx_DF.loc[i]
                    Rz_toappend = Rz_DF.loc[i]
                    Ry_toappend = Ry_DF.loc[i]                                                                            
                    
                    arr = []
                    for j in BERT_new[i][1:-1]:
                        bert.append(j[1])
                        arr.append(j[1])

                    for i in range(0, Nb_to_add):
                        F0_toappend = F0_toappend.append(rowF0)
                        AU1_toappend = AU1_toappend.append(rowAU)
                        AU2_toappend = AU2_toappend.append(rowAU)
                        AU4_toappend = AU4_toappend.append(rowAU)
                        AU5_toappend = AU5_toappend.append(rowAU)
                        AU6_toappend = AU6_toappend.append(rowAU)
                        AU7_toappend = AU7_toappend.append(rowAU)
                        Rx_toappend = Rx_toappend.append(rowAU)
                        Rz_toappend = Rz_toappend.append(rowAU)
                        Ry_toappend = Ry_toappend.append(rowAU)
                        arr.append(rowBERT)
                        
                    dict_bert[i] = arr

                    BERT.append(np.array(arr))
                    F0.append(F0_toappend.to_numpy())
                    AU1_in.append(AU1_toappend.to_numpy())
                    AU2_in.append(AU2_toappend.to_numpy())
                    AU4_in.append(AU4_toappend.to_numpy())
                    AU5_in.append(AU5_toappend.to_numpy())
                    AU6_in.append(AU6_toappend.to_numpy())
                    AU7_in.append(AU7_toappend.to_numpy())
                    RX_DF.append(Rx_toappend.to_numpy())
                    RZ_DF.append(Rz_toappend.to_numpy())
                    RY_DF.append(Ry_toappend.to_numpy())
                                                                                                                    

                if NbWords >= 10:
                    F0_toappend = DFAllData_F0.loc[i].head(10)
                    AU1_toappend = AU01_in.loc[i].head(10)
                    AU2_toappend = AU02_in.loc[i].head(10)
                    AU4_toappend = AU04_in.loc[i].head(10)
                    AU5_toappend = AU05_in.loc[i].head(10)
                    AU6_toappend = AU06_in.loc[i].head(10)
                    AU7_toappend = AU07_in.loc[i].head(10)
                    Rx_toappend = Rx_DF.loc[i].head(10)
                    Rz_toappend = Rz_DF.loc[i].head(10)
                    Ry_toappend = Ry_DF.loc[i].head(10)
                    
                    arr = []
                    for j in BERT_new[i][1:11]:
                        bert.append(j[1])
                        arr.append(j[1])
                    dict_bert[i] = arr
                    BERT.append(np.array(arr))
                    F0.append(F0_toappend.to_numpy())
                    AU1_in.append(AU1_toappend.to_numpy())
                    AU2_in.append(AU2_toappend.to_numpy())
                    AU4_in.append(AU4_toappend.to_numpy())
                    AU5_in.append(AU5_toappend.to_numpy())
                    AU6_in.append(AU6_toappend.to_numpy())
                    AU7_in.append(AU7_toappend.to_numpy())
                    RX_DF.append(Rx_toappend.to_numpy())
                    RZ_DF.append(Rz_toappend.to_numpy())
                    RY_DF.append(Ry_toappend.to_numpy())
                    
    F0 = np.array(F0)
    AU1_in = np.array(AU1_in)
    AU2_in = np.array(AU2_in)
    AU4_in = np.array(AU4_in)
    AU5_in = np.array(AU5_in)
    AU6_in = np.array(AU6_in)
    AU7_in = np.array(AU7_in)
    RX_DF = np.array(RX_DF)
    RZ_DF = np.array(RZ_DF)
    RY_DF = np.array(RY_DF)
    BERT = np.array(BERT)
    IPUs_filtered = np.array(IPUs_filtered)
    Speakers_Filtered = np.array(Speakers_Filtered)
    Speakers_Filtered_shaped = []

    for i in  Speakers_Filtered:
        arr = np.repeat(np.repeat(i, 100), 10)
        Speakers_Filtered_shaped.append(arr)

    Speakers_Filtered_shaped = np.array(Speakers_Filtered_shaped)
    Speakers_Filtered_shaped = np.reshape(Speakers_Filtered_shaped, (Speakers_Filtered_shaped.shape[0], 10, 100))

    return F0, AU1_in, AU2_in, AU4_in, AU5_in, AU6_in, AU7_in, RX_DF, RY_DF, RZ_DF, BERT, IPUs_filtered, Speakers_Filtered, Speakers_Filtered_shaped


def InferencePredictions(MODE, size, Speakers, F0, Bert, AU1, AU2, AU4, AU5, AU6, AU7, Rx, Ry, Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ):
    
 allpredictions1 = []
 allgroundtruth1 = []
 allpredictions2 = []
 allgroundtruth2 = []
 allpredictions3 = []
 allgroundtruth3 = []
 allpredictions4 = []
 allgroundtruth4 = []
 allpredictions5 = []
 allgroundtruth5 = []
 allpredictions6 = []
 allgroundtruth6 = []
 allpredictions7 = []
 allgroundtruth7 = []
 allpredictions9 = []
 allgroundtruth9 = []
 allpredictions8 = []
 allgroundtruth8 = []

 for i in range(size):
    trans1 = [1] #start token
    trans2 = [1] #start token
    trans3 = [1] #start token
    trans4 = [1] #start token
    trans5 = [1] #start token
    trans6 = [1] #start token
    trans7 = [1] #start token
    trans9 = [1] #start token
    trans8 = [1] #start token

    for _ in range(100):
        spk = np.reshape(Speakers[i: i+1], (1, 10, 100))
        Freq = np.reshape(F0[i:i+1], (1, 10, 100))
        Brt = np.reshape(Bert[i:i+1], (1, 10, 768))
        
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = IPU_level_model.predict([spk, Freq, Brt,  np.asarray([trans1]),   np.asarray([trans2]),   np.asarray([trans3]),   np.asarray([trans4]),   np.asarray([trans5]),   np.asarray([trans6]), np.asarray([trans7]),  np.asarray([trans8]), np.asarray([trans9])])
        trans1.append(np.argmax(p1[-1,-1]))
        trans2.append(np.argmax(p2[-1,-1]))
        trans3.append(np.argmax(p3[-1,-1]))
        trans4.append(np.argmax(p4[-1,-1]))
        trans5.append(np.argmax(p5[-1,-1]))
        trans6.append(np.argmax(p6[-1,-1]))
        trans7.append(np.argmax(p7[-1,-1]))
        trans8.append(np.argmax(p8[-1,-1]))
        trans9.append(np.argmax(p9[-1,-1]))

        if trans1[-1] == target_vocab_size + 1:
            break

    real_translation1 = []
    real_translation2 = []
    real_translation3 = []
    real_translation4 = []
    real_translation5 = []
    real_translation6 = []
    real_translation7 = []
    real_translation8 = []
    real_translation9 = []

    print('speaker is: ', spk[0][0][0])
    print('speaker is : ', ID_to_Speaker.get(spk[0][0][0]))


    for w in AU1[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation1.append(w)

    for w in AU2[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation2.append(w)

    for w in AU4[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation3.append(w)

    for w in AU5[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation4.append(w)

    for w in AU6[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation5.append(w)

    for w in AU7[i][1:]:
        if w == target_vocab_size + 1:
            break
        real_translation6.append(w)

    for w in Rx[i][1:]:
        if w == num_decoder_tokens_RX + 1:
            break
        real_translation7.append(w)

    for w in Ry[i][1:]:
        if w == num_decoder_tokens_RY + 1:
            break
        real_translation8.append(w)

        
    for w in Rz[i][1:]:
        if w == num_decoder_tokens_RZ + 1:
            break
        real_translation9.append(w)


    GN1 = IntegerDecode(ID_to_AU,(real_translation1)).tolist()
    Pred1 = IntegerDecode(ID_to_AU, (trans1[1:-1])).tolist()
    if 9.0 in GN1:
        indexOfEOS1 = GN1.index(9.0)
    else:
        indexOfEOS1 = len(GN1)
    new_raw_decoded1 = GN1[:indexOfEOS1]
    new_prediction = Pred1[:len(new_raw_decoded1)]
    new_prediction = [0.0 if x==9.0 else x for x in new_prediction]
    print('GT AU01:   ', new_raw_decoded1)
    print('Pred AU01: ', new_prediction)
    allpredictions1.append(new_prediction)
    allgroundtruth1.append(new_raw_decoded1)


    GN2 = IntegerDecode(ID_to_AU,(real_translation2)).tolist()
    Pred2 = IntegerDecode(ID_to_AU, (trans2[1:-1])).tolist()
    if 9.0 in GN2:
        indexOfEOS2 = GN2.index(9.0)
    else:
        indexOfEOS2 = len(GN2)
    new_raw_decoded2 = GN2[:indexOfEOS2]
    new_prediction2 = Pred2[:len(new_raw_decoded2)]
    new_prediction2 = [0.0 if x==9.0 else x for x in new_prediction2]
    print('GT AU02:   ', new_raw_decoded2)
    print('Pred AU02: ', new_prediction2)
    allpredictions2.append(new_prediction2)
    allgroundtruth2.append(new_raw_decoded2)

    GN3 = IntegerDecode(ID_to_AU,(real_translation3)).tolist()
    Pred3 = IntegerDecode(ID_to_AU, (trans3[1:-1])).tolist()
    if 9.0 in GN3:
        indexOfEOS3 = GN3.index(9.0)
    else:
        indexOfEOS3 = len(GN3)
    new_raw_decoded3 = GN3[:indexOfEOS3]
    new_prediction3 = Pred3[:len(new_raw_decoded3)]
    new_prediction3 = [0.0 if x==9.0 else x for x in new_prediction3]
    print('GT AU04:   ', new_raw_decoded3)
    print('Pred AU04: ', new_prediction3)
    allpredictions3.append(new_prediction3)
    allgroundtruth3.append(new_raw_decoded3)

    GN4 = IntegerDecode(ID_to_AU,(real_translation4)).tolist()
    Pred4 = IntegerDecode(ID_to_AU, (trans4[1:-1])).tolist()
    if 9.0 in GN4:
        indexOfEOS4 = GN4.index(9.0)
    else:
        indexOfEOS4 = len(GN4)
    new_raw_decoded4 = GN4[:indexOfEOS4]
    new_prediction4 = Pred4[:len(new_raw_decoded4)]
    new_prediction4 = [0.0 if x==9.0 else x for x in new_prediction4]
    print('GT AU05:   ', new_raw_decoded4)
    print('Pred AU05: ', new_prediction4)
    allpredictions4.append(new_prediction4)
    allgroundtruth4.append(new_raw_decoded4)

    GN5 = IntegerDecode(ID_to_AU,(real_translation5)).tolist()
    Pred5 = IntegerDecode(ID_to_AU, (trans5[1:-1])).tolist()
    if 9.0 in GN5:
        indexOfEOS5 = GN5.index(9.0)
    else:
        indexOfEOS5 = len(GN5)
    new_raw_decoded5 = GN5[:indexOfEOS5]
    new_prediction5 = Pred5[:len(new_raw_decoded5)]
    new_prediction5 = [0.0 if x==9.0 else x for x in new_prediction5]
    print('GT AU06:   ', new_raw_decoded5)
    print('Pred AU06: ', new_prediction5)
    allpredictions5.append(new_prediction5)
    allgroundtruth5.append(new_raw_decoded5)

    GN6 = IntegerDecode(ID_to_AU,(real_translation6)).tolist()
    Pred6 = IntegerDecode(ID_to_AU, (trans6[1:-1])).tolist()
    if 9.0 in GN6:
        indexOfEOS6 = GN6.index(9.0)
    else:
        indexOfEOS6 = len(GN6)
    new_raw_decoded6 = GN6[:indexOfEOS6]
    new_prediction6 = Pred6[:len(new_raw_decoded6)]
    new_prediction6 = [0.0 if x==9.0 else x for x in new_prediction6]
    print('GT AU07:   ', new_raw_decoded6)
    print('Pred AU07: ', new_prediction6)
    allpredictions6.append(new_prediction6)
    allgroundtruth6.append(new_raw_decoded6)
    print('  ')

    GN7 = IntegerDecode(ID_to_RX,(real_translation7)).tolist()
    Pred7 = IntegerDecode(ID_to_RX, (trans7[1:-1])).tolist()
    if 9.0 in GN7:
        indexOfEOS7 = GN7.index(9.0)
    else:
        indexOfEOS7 = len(GN7)
    new_raw_decoded7 = GN7[:indexOfEOS7]
    new_prediction7 = Pred7[:len(new_raw_decoded7)]
    new_prediction7 = [0.0 if x==9.0 else x for x in new_prediction7]
    print('GT RX:   ', new_raw_decoded7)
    print('Pred RX: ', new_prediction7)
    allpredictions7.append(new_prediction7)
    allgroundtruth7.append(new_raw_decoded7)
    print('  ')

    GN8 = IntegerDecode(ID_to_RY,(real_translation8)).tolist()
    Pred8 = IntegerDecode(ID_to_RY, (trans8[1:-1])).tolist()
    if 9.0 in GN8:
        indexOfEOS8 = GN8.index(9.0)
    else:
        indexOfEOS8 = len(GN8)
    new_raw_decoded8 = GN8[:indexOfEOS8]
    new_prediction8 = Pred8[:len(new_raw_decoded8)]
    new_prediction8 = [0.0 if x==9.0 else x for x in new_prediction8]
    print('GT RY:   ', new_raw_decoded8)
    print('Pred RY: ', new_prediction8)
    allpredictions8.append(new_prediction8)
    allgroundtruth8.append(new_raw_decoded8)
    print('  ')
    
    GN9 = IntegerDecode(ID_to_RZ,(real_translation9)).tolist()
    Pred9 = IntegerDecode(ID_to_RZ, (trans9[1:-1])).tolist()
    if 9.0 in GN9:
        indexOfEOS9 = GN9.index(9.0)
    else:
        indexOfEOS9 = len(GN9)
    new_raw_decoded9 = GN9[:indexOfEOS9]
    new_prediction9 = Pred9[:len(new_raw_decoded9)]
    new_prediction9 = [0.0 if x==9.0 else x for x in new_prediction9]
    print('GT RZ:   ', new_raw_decoded9)
    print('Pred RZ: ', new_prediction9)
    allpredictions9.append(new_prediction9)
    allgroundtruth9.append(new_raw_decoded9)
    print('  ')
    # saving each IPU alone
    '''
    zeros = np.zeros(len(new_prediction9))
    DF_GroundTruth = DataFrame({'timestamp':zeros, 'gaze_0_x':zeros, 'gaze_0_y':zeros, 'gaze_0_z':zeros, 'gaze_1_x':zeros, 'gaze_1_y':zeros, 'gaze_1_z':zeros, 'gaze_angle_x':zeros, 'gaze_angle_y':zeros, 'pose_Tx':zeros, 'pose_Ty':zeros, 'pose_Tz':zeros, 'pose_Rx':new_raw_decoded7, 'pose_Ry':zeros, 'pose_Rz':new_raw_decoded9, 'AU01_r':new_raw_decoded1, 'AU02_r':new_raw_decoded2, 'AU04_r':new_raw_decoded3, 'AU05_r':new_raw_decoded4, 'AU06_r':new_raw_decoded5, 'AU07_r':new_raw_decoded6, 'AU09_r':zeros, 'AU10_r':zeros, 'AU12_r':zeros, 'AU14_r':zeros, 'AU15_r':zeros, 'AU17_r':zeros, 'AU20_r':zeros, 'AU23_r':zeros, 'AU25_r':zeros, 'AU26_r':zeros, 'AU45_r':zeros})

    DF_Prediction = DataFrame({'timestamp':zeros, 'gaze_0_x':zeros, 'gaze_0_y':zeros, 'gaze_0_z':zeros, 'gaze_1_x':zeros, 'gaze_1_y':zeros, 'gaze_1_z':zeros, 'gaze_angle_x':zeros, 'gaze_angle_y':zeros, 'pose_Tx':zeros, 'pose_Ty':zeros, 'pose_Tz':zeros, 'pose_Rx':new_prediction7, 'pose_Ry':zeros, 'pose_Rz':new_prediction9, 'AU01_r':new_prediction, 'AU02_r':new_prediction2, 'AU04_r':new_prediction3, 'AU05_r':new_prediction4, 'AU06_r':new_prediction5, 'AU07_r':new_prediction6, 'AU09_r':zeros, 'AU10_r':zeros, 'AU12_r':zeros, 'AU14_r':zeros, 'AU15_r':zeros, 'AU17_r':zeros, 'AU20_r':zeros, 'AU23_r':zeros, 'AU25_r':zeros, 'AU26_r':zeros, 'AU45_r':zeros})

    DF_GroundTruth.to_csv(MODE+"_"+ID_to_Speaker.get(spk[0][0][0])+"_"+str(i)+"_Raw.csv", index=False)
    DF_Prediction.to_csv(MODE+"_"+ID_to_Speaker.get(spk[0][0][0])+"_"+str(i)+"_Pred.csv", index=False)
    '''


 # saving successive IPUs together
 allpredictions1 = convert2DListto1DList(allpredictions1)
 allpredictions2 = convert2DListto1DList(allpredictions2)
 allpredictions3 = convert2DListto1DList(allpredictions3)
 allpredictions4 = convert2DListto1DList(allpredictions4)
 allpredictions5 = convert2DListto1DList(allpredictions5)
 allpredictions6 = convert2DListto1DList(allpredictions6)
 allpredictions7 = convert2DListto1DList(allpredictions7)
 allpredictions8 = convert2DListto1DList(allpredictions8)
 allpredictions9 = convert2DListto1DList(allpredictions9)

 #median filter
 allpredictions1 = signal.medfilt(allpredictions1, 7)
 allpredictions2 = signal.medfilt(allpredictions2, 7)
 allpredictions3 = signal.medfilt(allpredictions3, 7)
 allpredictions4 = signal.medfilt(allpredictions4, 7)
 allpredictions5 = signal.medfilt(allpredictions5, 7)
 allpredictions6 = signal.medfilt(allpredictions6, 7)
 allpredictions7 = signal.medfilt(allpredictions7, 7)
 allpredictions8 = signal.medfilt(allpredictions8, 7)
 allpredictions9 = signal.medfilt(allpredictions9, 7)
 
 zeros = np.zeros(len(allpredictions9))
 
 allgroundtruth1 = convert2DListto1DList(allgroundtruth1)
 allgroundtruth2 = convert2DListto1DList(allgroundtruth2)
 allgroundtruth3 = convert2DListto1DList(allgroundtruth3)
 allgroundtruth4 = convert2DListto1DList(allgroundtruth4)
 allgroundtruth5 = convert2DListto1DList(allgroundtruth5)
 allgroundtruth6 = convert2DListto1DList(allgroundtruth6)
 allgroundtruth7 = convert2DListto1DList(allgroundtruth7)
 allgroundtruth8 = convert2DListto1DList(allgroundtruth8)
 allgroundtruth9 = convert2DListto1DList(allgroundtruth9)
 
 #median filter
 allgroundtruth1 = signal.medfilt(allgroundtruth1, 7)
 allgroundtruth2 = signal.medfilt(allgroundtruth2, 7)
 allgroundtruth3 = signal.medfilt(allgroundtruth3, 7)
 allgroundtruth4 = signal.medfilt(allgroundtruth4, 7)
 allgroundtruth5 = signal.medfilt(allgroundtruth5, 7)
 allgroundtruth6 = signal.medfilt(allgroundtruth6, 7)
 allgroundtruth7 = signal.medfilt(allgroundtruth7, 7)
 allgroundtruth8 = signal.medfilt(allgroundtruth8, 7)
 allgroundtruth9 = signal.medfilt(allgroundtruth9, 7)


 CalculateMetrics(allpredictions1, allgroundtruth1, 1, MODE)
 CalculateMetrics(allpredictions2, allgroundtruth2, 2, MODE)
 CalculateMetrics(allpredictions3, allgroundtruth3, 3, MODE)
 CalculateMetrics(allpredictions4, allgroundtruth4, 4, MODE)
 CalculateMetrics(allpredictions5, allgroundtruth5, 5, MODE)
 CalculateMetrics(allpredictions6, allgroundtruth6, 6, MODE)
 CalculateMetrics(allpredictions7, allgroundtruth7, 7, MODE)
 CalculateMetrics(allpredictions8, allgroundtruth8, 8, MODE)
 CalculateMetrics(allpredictions9, allgroundtruth9, 9, MODE)

 #rescaling 1/2
 allpredictions1 = allpredictions1/2
 allpredictions2 = allpredictions2/2
 allpredictions3 = allpredictions3/2
 allpredictions4 = allpredictions4/2
 allpredictions5 = allpredictions5/2
 allpredictions6 = allpredictions6/2
 allpredictions7 = allpredictions7/2
 allpredictions8 = allpredictions8/2
 allpredictions9 = allpredictions9/2

 allgroundtruth1 = allgroundtruth1/2
 allgroundtruth2 = allgroundtruth2/2
 allgroundtruth3 = allgroundtruth3/2
 allgroundtruth4 = allgroundtruth4/2
 allgroundtruth5 = allgroundtruth5/2
 allgroundtruth6 = allgroundtruth6/2
 allgroundtruth7 = allgroundtruth7/2
 allgroundtruth8 = allgroundtruth8/2
 allgroundtruth9 = allgroundtruth9/2

 DF_Prediction = DataFrame({'timestamp':zeros, 'gaze_0_x':zeros, 'gaze_0_y':zeros, 'gaze_0_z':zeros, 'gaze_1_x':zeros, 'gaze_1_y':zeros, 'gaze_1_z':zeros, 'gaze_angle_x':zeros, 'gaze_angle_y':zeros, 'pose_Tx':zeros, 'pose_Ty':zeros, 'pose_Tz':zeros, 'pose_Rx':allpredictions7, 'pose_Ry':allpredictions8, 'pose_Rz':allpredictions9, 'AU01_r':allpredictions1, 'AU02_r':allpredictions2, 'AU04_r':allpredictions3, 'AU05_r':allpredictions4, 'AU06_r':allpredictions5, 'AU07_r': allpredictions6, 'AU09_r':zeros, 'AU10_r':zeros, 'AU12_r':zeros, 'AU14_r':zeros, 'AU15_r':zeros, 'AU17_r':zeros, 'AU20_r':zeros, 'AU23_r':zeros, 'AU25_r':zeros, 'AU26_r':zeros, 'AU45_r':zeros})

 DF_GroundTruth = DataFrame({'timestamp':zeros, 'gaze_0_x':zeros, 'gaze_0_y':zeros, 'gaze_0_z':zeros, 'gaze_1_x':zeros, 'gaze_1_y':zeros, 'gaze_1_z':zeros,'gaze_angle_x':zeros, 'gaze_angle_y':zeros, 'pose_Tx':zeros, 'pose_Ty':zeros, 'pose_Tz':zeros, 'pose_Rx':allgroundtruth7, 'pose_Ry':allgroundtruth8, 'pose_Rz':allgroundtruth9, 'AU01_r':allgroundtruth1, 'AU02_r':allgroundtruth2, 'AU04_r':allgroundtruth3, 'AU05_r':allgroundtruth4, 'AU06_r':allgroundtruth5, 'AU07_r':allgroundtruth6, 'AU09_r':zeros, 'AU10_r':zeros, 'AU12_r':zeros, 'AU14_r':zeros, 'AU15_r':zeros, 'AU17_r':zeros, 'AU20_r':zeros, 'AU23_r':zeros, 'AU25_r':zeros, 'AU26_r':zeros, 'AU45_r':zeros})

 DF_GroundTruth.to_csv(MODE+"_"+ID_to_Speaker.get(spk[0][0][0])+"_Raw.csv", index=False)
 DF_Prediction.to_csv(MODE+"_"+ID_to_Speaker.get(spk[0][0][0])+"_Pred.csv", index=False)




 
# Hyperparameters
pd.options.display.float_format = '{:.4f}'.format
BATCH_SIZE = 32
BUFFER_SIZE = 20000



#------------------------------------------------ data preprocessing ---------------------------------------------
f1 = open('/u/anasynth/fares/Seq2Seq/IDs/IDs_2.txt')
##f1 = open('/u/anasynth/fares/Seq2Seq/IDs/IDs_test.txt')
##f1 = open('/u/anasynth/fares/Seq2Seq/IDs/IDs_SpeakerStyles.txt')
##f1 = open('/u/anasynth/fares/Seq2Seq/IDs/IDs_1687.txt')

MinRx_ = 0
MinRy_ = 0
MinRz_ = 0
MaxRx_ = 0
MaxRy_ = 0
MaxRz_ = 0
IPUs_filtered_ = []
Speakers_ = []
Speakers_filtered_ = []
NumberOfSpeakers = 0

for filename in f1 :
    print(filename)
    filename = filename.rstrip()
    F0_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE_F0_NonNormalized/PreprocessedF0_"+ filename+".p"
    AU_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE__AU_Final/AUPreprocessed_"+ filename+".p"
    IPU_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE_IPU_01082021/IPUs_"+ filename+".p"
    BERT_path = "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE_BERT/IPU_BERT_"+filename+".p"
    
    isExistF0 = os.path.exists(F0_path)
    if isExistF0:
        NumberOfSpeakers = NumberOfSpeakers + 1    
    
        BERT_DF, IPUNumber, IPU_DF, DFAllData_AU01, DFAllData_AU02, DFAllData_AU04, DFAllData_AU05, DFAllData_AU06, DFAllData_AU07, Tx, Ty, Tz, Rx, Ry, Rz, DFAllData_F0, VOCAB_F0 =  PreprocessData(F0_path, AU_path, IPU_path, BERT_path, filename)

        #Getting Min and Max of Head Position and Rotation
        RXX = np.array(Rx)
        RYY = np.array(Ry)
        RZZ = np.array(Rz)
        RXX = RXX[RXX != 7.0]
        RYY = RYY[RYY != 7.0]
        RZZ = RZZ[RZZ != 7.0]
        MinRx = (np.min(np.array(RXX)))
        if MinRx < MinRx_:
            MinRx_ = MinRx
        MinRy = (np.min(np.array(RYY)))
        if MinRy < MinRy_:
            MinRy_ = MinRy                    
        MinRz = (np.min(np.array(RZZ)))
        if MinRz < MinRz_:
            MinRz_ = MinRz                    
        MaxRx = (np.max(np.array(RXX)))
        if MaxRx > MaxRx_:
            MaxRx_ = MaxRx
        MaxRy = (np.max(np.array(RYY)))
        if MaxRy > MaxRy_:
            MaxRy_ = MaxRy
        MaxRz = (np.max(np.array(RZZ)))
        if MaxRz > MaxRz_:
            MaxRz_ = MaxRz


        IPUs = IPU_DF
        BERT_new = []
        BERT_DF = np.array(BERT_DF)
        
        for count, i in enumerate(BERT_DF):
            bbert = [x for x in i if str(x) != 'nan']
            bbert = [x for x in bbert if str(x) != 'None']
            BERT_new.append(bbert)
        print('1')    
        DFAllData_F0, AU01_in_DF, AU02_in_DF, AU04_in_DF, AU05_in_DF, AU06_in_DF, AU07_in_DF, Rx_DF, Ry_DF, Rz_DF = format_data(DFAllData_F0, DFAllData_AU01, DFAllData_AU02, DFAllData_AU04, DFAllData_AU05, DFAllData_AU06, DFAllData_AU07, Rx, Ry, Rz)
        print('2')
        F0, AU01_in, AU02_in, AU04_in, AU05_in, AU06_in, AU07_in, Rx_, Ry_, Rz_, BERT, IPUs_filtered, Speakers, Speakers_filtered = format_IPULevel(IPUs, DFAllData_F0,AU01_in_DF, AU02_in_DF, AU04_in_DF, AU05_in_DF, AU06_in_DF, AU07_in_DF, Rx_DF, Ry_DF, Rz_DF, BERT_new)

        AU1_in = TurnIPU_MergedWords(AU01_in)
        AU2_in = TurnIPU_MergedWords(AU02_in)
        AU4_in = TurnIPU_MergedWords(AU04_in)
        AU5_in = TurnIPU_MergedWords(AU05_in)
        AU6_in = TurnIPU_MergedWords(AU06_in)
        AU7_in = TurnIPU_MergedWords(AU07_in)
        Rx = TurnIPU_MergedWords(Rx_)
        Rz = TurnIPU_MergedWords(Rz_)
        Ry = TurnIPU_MergedWords(Ry_)
     
        if NumberOfSpeakers == 1:
            F0_ = F0
            AU01_in_ = AU1_in
            AU02_in_ = AU2_in
            AU04_in_ = AU4_in
            AU05_in_ = AU5_in
            AU06_in_ = AU6_in
            AU07_in_ = AU7_in
            Rx__ = Rx
            Ry__ = Ry
            Rz__ = Rz
            BERT_ = BERT
            IPUs_filtered_ = IPUs_filtered.tolist()
            Speakers_ = Speakers.tolist()
            Speakers_filtered_ = Speakers_filtered.tolist()

            
        else:
            print('3')
            F0_ = np.concatenate((F0_, np.array(F0)), axis=0)
            AU01_in_ = np.concatenate((AU01_in_, np.array(AU1_in)), axis=0)
            AU02_in_ = np.concatenate((AU02_in_, np.array(AU2_in)), axis=0)
            AU04_in_ = np.concatenate((AU04_in_, np.array(AU4_in)), axis=0)
            AU05_in_ = np.concatenate((AU05_in_, np.array(AU5_in)), axis=0)
            AU06_in_ = np.concatenate((AU06_in_, np.array(AU6_in)), axis=0)
            AU07_in_ = np.concatenate((AU07_in_, np.array(AU7_in)), axis=0)
            Rx__ = np.concatenate((Rx__, np.array(Rx)), axis=0)
            Ry__ = np.concatenate((Ry__, np.array(Ry)), axis=0)
            Rz__ = np.concatenate((Rz__, np.array(Rz)), axis=0)            
            BERT_ = np.concatenate((BERT_, np.array(BERT)), axis=0)
            for ipu in IPUs_filtered:
                IPUs_filtered_.append(ipu)
            for spk in  Speakers:
                Speakers_.append(spk)
            for s in Speakers_filtered:
                Speakers_filtered_.append(s)


device = get_comp_device()
print('device : ', device)

IPUs_filtered_ = np.array(IPUs_filtered_)
Speakers_filtered_ = np.array(Speakers_filtered_)
Speakers_ = np.array(Speakers_)
print('Speakers_filtered_: ', Speakers_filtered_.shape)
print('IPUs_filtered_: ', IPUs_filtered_.shape)
print('Speakers_: ', Speakers_.shape)


#Speakers vocab
SpeakersIDS = list(set((Speakers_)))
ID_to_Speaker = {}
for i in range(0, len(SpeakersIDS)):
    ID_to_Speaker[i] = SpeakersIDS[i]
Speaker_to_ID ={v:k for k, v in ID_to_Speaker.items()}    


#AU Quantization
quantization_step_AU = 0.01
minAU = 0
maxAU = 5.0
quant = np.arange(minAU, maxAU, quantization_step_AU)
ID_to_AU, AU_to_ID = buildVocab(quant)
AU_to_ID = {float("{0:.4f}".format(v)):k for k, v in ID_to_AU.items()}
ID_to_AU = {v:k for k, v in AU_to_ID.items()}

#F0 MinMax Scaling and Quantization
'''
scaler = MinMaxScaler()
shape1 = F0_.shape[0]
shape2 = F0_.shape[1]
shape3 = F0_.shape[2]
F0 = F0_.flatten()
F0 = np.array(F0).reshape(-1, 1)
scaler.fit(F0)
F0 = scaler.transform(F0)
quantization_step_F0 = 0.002
'''
minF0 = 50
maxF0 = 550
quantization_step_F0 = 1
quantF0 = np.arange(minF0, maxF0, quantization_step_F0)
F0 = F0Quantization(F0, quantF0)
F0 = np.array(F0)
print(F0)
print(F0.shape)
'''F0 = np.reshape(F0, (shape1, shape2, shape3))'''
ID_to_F0, F0_to_ID = buildVocabF0(quantF0)
print(stop)

#Head Position and Rotation quantization
quantization_step_Rx = 0.003
quantization_step_Ry = 0.002
quantization_step_Rz = 0.004
quantRx = np.arange(MinRx_, MaxRx_, quantization_step_Rx)
quantRy = np.arange(MinRy_, MaxRy_, quantization_step_Ry)
quantRz = np.arange(MinRz_, MaxRz_, quantization_step_Rz)
ID_to_RX, RX_to_ID = buildVocab(quantRx)
ID_to_RY, RY_to_ID = buildVocab(quantRy)
ID_to_RZ, RZ_to_ID = buildVocab(quantRz)
RX_to_ID = {float("{0:.4f}".format(v)):k for k, v in ID_to_RX.items()}
RY_to_ID = {float("{0:.4f}".format(v)):k for k, v in ID_to_RY.items()}
RZ_to_ID = {float("{0:.4f}".format(v)):k for k, v in ID_to_RZ.items()}
ID_to_RX = {v:k for k, v in RX_to_ID.items()}
ID_to_RY = {v:k for k, v in RY_to_ID.items()}
ID_to_RZ = {v:k for k, v in RZ_to_ID.items()}


num_speaker_encoder_tokens = len(Speaker_to_ID)
num_encoder_tokens = len(F0_to_ID)
num_decoder_tokens = len(AU_to_ID)
num_decoder_tokens_RX = len(RX_to_ID)
num_decoder_tokens_RY = len(RY_to_ID)
num_decoder_tokens_RZ = len(RZ_to_ID)
max_encoder_seq_length = (F0).shape[1]
max_decoder_seq_length = AU01_in_.shape[1]
max_number_word_per_IPU = 10
print('num_speaker_encoder_tokens: ', num_speaker_encoder_tokens)
print('num_encoder_tokens: ', num_encoder_tokens)
print('num_decoder_tokens: ', num_decoder_tokens)
print('num_decoder_tokens_RX: ', num_decoder_tokens_RX)
print('num_decoder_tokens_RZ: ', num_decoder_tokens_RZ)
print('max_encoder_seq_length: ', max_encoder_seq_length)
print('max_decoder_seq_length: ', max_decoder_seq_length)



#split into 80% train set, 10% validate set, 10% test set
AU1_train, AU2_train, AU4_train, AU5_train, AU6_train, AU7_train, F0train, F0validate, AU1_validate,AU2_validate, AU4_validate, AU5_validate, AU6_validate, AU7_validate, Bert_train, Bert_validate, Rx_train, Ry_train, Rz_train, Rx_validate, Ry_validate,  Rz_validate, Speakers_train, Speakers_validate, Successive1_F0, Successive2_F0, Successive1_AU1, Successive1_AU2, Successive1_AU4, Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive2_AU1, Successive2_AU2, Successive2_AU4, Successive2_AU5, Successive2_AU6, Successive2_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, Successive2_Rx, Successive2_Ry, Successive2_Rz, Successive1_Speakers, Successive2_Speakers, Successive2_bert, Successive1_bert, Successive1_IPUs, Successive2_IPUs = split_train_validate_test(F0, AU01_in_, AU02_in_, AU04_in_, AU05_in_, AU06_in_, AU07_in_, Rx__, Ry__, Rz__, BERT_, IPUs_filtered_, Speakers_filtered_)


#integer encoding
Successive1_F0 = IntegerEncodeF0(F0_to_ID, Successive1_F0)
Successive1_AU1 = IntegerEncodeAU(AU_to_ID, Successive1_AU1)
Successive1_AU2 = IntegerEncodeAU(AU_to_ID, Successive1_AU2)
Successive1_AU4 = IntegerEncodeAU(AU_to_ID, Successive1_AU4)
Successive1_AU5 = IntegerEncodeAU(AU_to_ID, Successive1_AU5)
Successive1_AU6 = IntegerEncodeAU(AU_to_ID, Successive1_AU6)
Successive1_AU7 = IntegerEncodeAU(AU_to_ID, Successive1_AU7)
Successive1_Rx = IntegerEncodeAU(RX_to_ID, Successive1_Rx)
Successive1_Ry = IntegerEncodeAU(RY_to_ID, Successive1_Ry)
Successive1_Rz = IntegerEncodeAU(RX_to_ID, Successive1_Rz)
Successive1_Speakers = IntegerEncodeAU(Speaker_to_ID, Successive1_Speakers)

Successive2_F0 = IntegerEncodeF0(F0_to_ID, Successive2_F0)
Successive2_AU1 = IntegerEncodeAU(AU_to_ID, Successive2_AU1)
Successive2_AU2 = IntegerEncodeAU(AU_to_ID, Successive2_AU2)
Successive2_AU4 = IntegerEncodeAU(AU_to_ID, Successive2_AU4)
Successive2_AU5 = IntegerEncodeAU(AU_to_ID, Successive2_AU5)
Successive2_AU6 = IntegerEncodeAU(AU_to_ID, Successive2_AU6) 
Successive2_AU7 = IntegerEncodeAU(AU_to_ID, Successive2_AU7)
Successive2_Rx = IntegerEncodeAU(RX_to_ID, Successive2_Rx)
Successive2_Ry = IntegerEncodeAU(RX_to_ID, Successive2_Ry)
Successive2_Rz = IntegerEncodeAU(RX_to_ID, Successive2_Rz)
Successive2_Speakers = IntegerEncodeAU(Speaker_to_ID, Successive2_Speakers)

AU1_train = IntegerEncodeAU(AU_to_ID, AU1_train)
AU2_train = IntegerEncodeAU(AU_to_ID, AU2_train)
AU4_train = IntegerEncodeAU(AU_to_ID, AU4_train)
AU5_train = IntegerEncodeAU(AU_to_ID, AU5_train)
AU6_train = IntegerEncodeAU(AU_to_ID, AU6_train)
AU7_train = IntegerEncodeAU(AU_to_ID, AU7_train)
Rx_train = IntegerEncodeAU(RX_to_ID, Rx_train)
Ry_train = IntegerEncodeAU(RY_to_ID, Ry_train)
Rz_train = IntegerEncodeAU(RZ_to_ID, Rz_train)
Speakers_train = IntegerEncodeAU(Speaker_to_ID, Speakers_train)
F0train = IntegerEncodeF0(F0_to_ID, F0train)

F0validate = IntegerEncodeF0(F0_to_ID, F0validate)
AU1_validate = IntegerEncodeAU(AU_to_ID, AU1_validate)
AU2_validate = IntegerEncodeAU(AU_to_ID, AU2_validate)
AU4_validate = IntegerEncodeAU(AU_to_ID, AU4_validate)
AU5_validate = IntegerEncodeAU(AU_to_ID, AU5_validate)
AU6_validate = IntegerEncodeAU(AU_to_ID, AU6_validate)
AU7_validate = IntegerEncodeAU(AU_to_ID, AU7_validate)
Rx_validate = IntegerEncodeAU(RX_to_ID, Rx_validate)
Ry_validate = IntegerEncodeAU(RY_to_ID, Ry_validate)
Rz_validate = IntegerEncodeAU(RZ_to_ID, Rz_validate)
Speakers_validate = IntegerEncodeAU(Speaker_to_ID, Speakers_validate)
ALLSpeakers= np.array(IntegerEncodeAU(Speaker_to_ID, Speakers_filtered_))
print(ALLSpeakers)


print('------------------------ VOCABS --------------------------------------------------------')
print('ID_to_F0: ', ID_to_F0)
print(' ')
print('ID_to_AU: ', ID_to_AU)
print(' ')
print('ID_to_RX: ', ID_to_RX)
print(' ')
print('ID_to_RZ: ', ID_to_RZ)
print(' ')
print('ID_to_RY: ', ID_to_RY)
print(' ')
print(' -------------------------- Dataset Format ------------------------------------------------------------------')
print('Speakers_train: ', Speakers_train.shape)
print('F0train : ', F0train.shape)
print('Bert_train: ', Bert_train.shape)
print('AU1in_train.shape: ', AU1_train.shape)
print('AU2_train: ', AU2_train.shape)
print('AU4_train: ', AU4_train.shape)
print('AU5_train: ', AU5_train.shape)
print('AU6_train: ', AU6_train.shape)
print('AU7_train: ', AU7_train.shape)
print('Rx_train: ', Rx_train.shape)
print('Rz_train: ', Rz_train.shape)
print('F0_train: ', F0train.shape)
print(' ')
print('Speakers_validate: ', Speakers_validate.shape)
print('Bert_validate: ',Bert_validate.shape)
print('F0_validate: ', F0validate.shape)
print('AU1in_validate: ', AU1_validate.shape)
print('AU2in_validate: ', AU2_validate.shape)
print('AU4in_validate: ', AU4_validate.shape)
print('AU5in_validate: ', AU5_validate.shape)
print('AU6in_validate: ', AU6_validate.shape)
print('AU7in_validate: ', AU7_validate.shape)
print('Rx_validate: ', Rx_validate.shape)
print('Ry_validate: ', Ry_validate.shape)
print('Rz_validate: ', Rz_validate.shape)
print(' -----------------------------------------------------------------------------------------------------')


#creating dataset
dataset_train = tf.data.Dataset.from_tensor_slices((Speakers_train, F0train, Bert_train, AU1_train, AU2_train, AU4_train, AU5_train, AU6_train, AU7_train, Rx_train, Ry_train, Rz_train))
dataset_validate = tf.data.Dataset.from_tensor_slices((Speakers_validate, F0validate, Bert_validate, AU1_validate, AU2_validate, AU4_validate, AU5_validate, AU6_validate, AU7_validate, Rx_validate, Ry_validate, Rz_validate))

dataset_train = dataset_train.map(tf_encode)
dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

dataset_validate = dataset_validate.map(tf_encode)
dataset_validate = dataset_validate.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)

print('training dataset: ', dataset_train)
print(' ')
print('dataset_validate: ', dataset_validate)
print(' ')
print('-------------------------------------------------------------------------------------------------')



'''------  COMPLETE MODEL ------ '''
#Hyperparameters
d_model = 64
num_heads_enc = 4
num_heads = 2
dff = 128
num_enc = 4
num_dec = 1
dropout = 0.1
input_vocab_size = num_encoder_tokens
target_vocab_size = num_decoder_tokens
 
SpeakerInput = Input(shape=(None, 100))
IPU_level_F0 = Input(shape=(None, 100))
IPU_level_Bert = Input(shape=(None, 768))
Bert_repeated = TimeDistributed(RepeatVector(100))(IPU_level_Bert)
target1 = Input(shape=(None, ))
target2 = Input(shape=(None, ))
target3 = Input(shape=(None, ))
target4 = Input(shape=(None, ))
target5 = Input(shape=(None, ))
target6 = Input(shape=(None, ))
Rx = Input(shape=(None, ))
Rz = Input(shape=(None, ))
Ry = Input(shape=(None, ))

FF0Encoder = F0Encoder(num_encoder_tokens, num_layers = num_enc, d_model = d_model, num_heads = num_heads_enc, dff = dff, dropout = dropout)
BERTEncoder = BertEncoder()
decoder = Decoder(num_decoder_tokens, num_layers = num_dec, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)
decoderRX = Decoder(num_decoder_tokens_RX, num_layers = num_dec, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)
decoderRZ = Decoder(num_decoder_tokens_RZ, num_layers = num_dec, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)
decoderRY = Decoder(num_decoder_tokens_RY, num_layers = num_dec, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)

SelfAttention = MultiHeadAttention(d_model, num_enc)
#Speaker embeddings
SpeakerEmb_ID = TimeDistributed(tf.keras.layers.Embedding(num_speaker_encoder_tokens, d_model))(SpeakerInput)
SpeakerEmb_F0 = SelfAttention([IPU_level_F0, IPU_level_F0, IPU_level_F0], mask = [FF0Encoder.compute_mask(IPU_level_F0), FF0Encoder.compute_mask(IPU_level_F0)])
SpeakerEmb_AU1 = SelfAttention([target1, target1, target1]) 
SpeakerEmb_AU2 = SelfAttention([target2, target2, target2])
SpeakerEmb_AU4 = SelfAttention([target3, target3, target3])
SpeakerEmb_AU124 = SelfAttention([SpeakerEmb_AU1, SpeakerEmb_AU2, SpeakerEmb_AU4])
SpeakerEmb_RX = SelfAttention([Rx, Rx, Rx])
SpeakerEmb_RY = SelfAttention([Ry, Ry, Ry])
SpeakerEmb_RZ = SelfAttention([Rz, Rz, Rz])
SpeakerEmb_RXYZ = SelfAttention([SpeakerEmb_RX, SpeakerEmb_RY, SpeakerEmb_RZ])

SpeakerEmb = tf.keras.layers.Concatenate()([SpeakerEmb_AU124, SpeakerEmb_RXYZ])
SpeakerEmb = tf.keras.layers.Concatenate()([SpeakerEmb, SpeakerEmb_ID])

SpeakerModel = Model(inputs=[SpeakerInput], outputs=[SpeakerEmb])

#model for F0 WORD level encoding
Word_level_F0 = Input(shape=(100, ))
x = FF0Encoder(Word_level_F0)
F0EncoderModel = tf.keras.models.Model(inputs=Word_level_F0, outputs= x)

#model for BERT WORD level encoding
Word_level_BERT = Input(shape=(None, ))
x = BERTEncoder(Word_level_BERT)
BERTEncoderModel = tf.keras.models.Model(inputs=Word_level_BERT, outputs= x)

F0_encoded = TimeDistributed(F0EncoderModel)(IPU_level_F0)
BERT_encoded = TimeDistributed(BERTEncoderModel)(Bert_repeated)

TransformerDecoder = BERTDecoder()
CrossAttDecoder = TransformerDecoder([F0_encoded, BERT_encoded], mask = FF0Encoder.compute_mask(IPU_level_F0))
CrossAttDecoder = tf.keras.layers.Concatenate()([SpeakerEmb, CrossAttDecoder])


A1 = decoder([target1,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
A2 = decoder([target2,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
A4 = decoder([target3,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
A5 = decoder([target4,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
A6 = decoder([target5,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
A7 = decoder([target6,  CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
RX = decoderRX([Rx, CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
RZ = decoderRZ([Rz, CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))
RY = decoderRY([Ry, CrossAttDecoder], mask=FF0Encoder.compute_mask(IPU_level_F0))

out1 = Dense(num_decoder_tokens, name = 'AU1')(A1)
out2 = Dense(num_decoder_tokens, name = 'AU2')(A2)
out4 = Dense(num_decoder_tokens, name = 'AU4')(A4)
out5 = Dense(num_decoder_tokens, name = 'AU5')(A5)
out6 = Dense(num_decoder_tokens, name = 'AU6')(A6)
out7 = Dense(num_decoder_tokens, name = 'AU7')(A7)
out8 = Dense(num_decoder_tokens_RX, name = 'Rx')(RX)
out10 = Dense(num_decoder_tokens_RZ, name = 'Rz')(RZ)
out9 = Dense(num_decoder_tokens_RY, name = 'Ry')(RY)

IPU_level_model = tf.keras.models.Model(inputs=[SpeakerInput, IPU_level_F0, IPU_level_Bert, target1, target2, target3, target4, target5, target6, Rx, Ry, Rz],outputs= [out1, out2, out4, out5, out6, out7, out8, out9, out10])
optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')




def masked_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    _loss = loss(y_true, y_pred)
    mask = tf.cast(mask, dtype=_loss.dtype)
    _loss *= mask
    return tf.reduce_sum(_loss)/tf.reduce_sum(mask)

metrics = [loss]
IPU_level_model.compile(optimizer= optimizer, loss = loss, metrics = metrics)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 300)

num_batches = len(dataset_train)
val_batches = len(dataset_validate)
                    

'''when fitting we want input, target as inputs and target as an output. we need the target_input to be shifted 1 from the target_out'''
def generator(dataset):
    while True:
        for Speaker, F0, bert, AU1, AU2, AU4, AU5, AU6, AU7, Rx, Ry, Rz in dataset:
            yield ([Speaker, F0, bert, AU1[:, :-1],  AU2[:, :-1], AU4[:,:-1], AU5[:,:-1], AU6[:,:-1], AU7[:,:-1], Rx[:,:-1], Ry[:,:-1], Rz[:,:-1]], [AU1[:,1:], AU2[:,1:],AU4[:,1:], AU5[:,1:], AU6[:,1:], AU7[:,1:], Rx[:,1:], Ry[:,1:], Rz[:,1:]])
                                    



with tf.device(device):
    history = IPU_level_model.fit(x = generator(dataset_train), validation_data = generator(dataset_validate), epochs=500, steps_per_epoch = num_batches, validation_steps = val_batches, callbacks = [callback]).history

    
# get speaker embeddings
SPKEmbeddings = []
SPK = []
for i in range(len(ALLSpeakers)):
    spk = np.reshape(ALLSpeakers[i: i+1], (1, 10, 100))
    spk_embedding = SpeakerModel.predict([spk])
    SPK.append(ID_to_Speaker.get(spk[0][0][0]))
    spk_embedding = np.reshape(spk_embedding, (spk_embedding.shape[0], spk_embedding.shape[1]*spk_embedding.shape[2]*spk_embedding.shape[3]))
    SPKEmbeddings.append(spk_embedding)


Plot_Losses(history)
Plot_TSNE(SPKEmbeddings, SPK)
Plot_PCA(SPKEmbeddings, SPK)


''' Prediction'''
'''
When predicting input what you want to translate to the encoder and input the start token to the decoder.
Repeat this except use the output of the last prediction as the input to the decoder.
Stop when the last value of the output is the stop token.
This final output (with the start and stop tokens removed) can be fed to tokenizer_en.decode() to get an english sentence
'''

#InferencePredictions("TRAIN", 6, Speakers_train, F0train, Bert_train, AU1_train, AU2_train, AU4_train, AU5_train, AU6_train, AU7_train, Rx_train, Rz_train, ID_to_AU, ID_to_RX, ID_to_RZ)
right_speaker = ID_to_Speaker.get(Successive1_Speakers[0][0][0])
print(right_speaker)
InferencePredictions("SUCCESSIVE1_RIGHT_"+right_speaker+"_", len(Successive1_F0), Successive1_Speakers, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4, Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)

arr_mGbMwP8MDjg = np.array(GetSpeakerArray('mGbMwP8MDjg', Speaker_to_ID))
arr_qQ_PUXPVlos = np.array(GetSpeakerArray('qQ-PUXPVlos', Speaker_to_ID))
arr_HNMQ_w7hXTA = np.array(GetSpeakerArray('HNMQ_w7hXTA', Speaker_to_ID))
arr_akiQuyhXR8o = np.array(GetSpeakerArray('akiQuyhXR8o', Speaker_to_ID))
arr_XAcARiiK5uY = np.array(GetSpeakerArray('XAcARiiK5uY', Speaker_to_ID))
arr_qtcWebAYmKY = np.array(GetSpeakerArray('qtcWebAYmKY', Speaker_to_ID))
arr_0d6iSvF1UmA = np.array(GetSpeakerArray('0d6iSvF1UmA', Speaker_to_ID))
arr__x1qkuvUxuI = np.array(GetSpeakerArray('_x1qkuvUxuI', Speaker_to_ID))
arr_NXfYNdapq3Q = np.array(GetSpeakerArray('NXfYNdapq3Q', Speaker_to_ID))
arr__MBiP3G2Pzc = np.array(GetSpeakerArray('_MBiP3G2Pzc', Speaker_to_ID))
arr_xHHb7R3kx40 = np.array(GetSpeakerArray('xHHb7R3kx40', Speaker_to_ID))
arr_apbSsILLh28 = np.array(GetSpeakerArray('apbSsILLh28', Speaker_to_ID))
arr_yBFC_RtfTfg = np.array(GetSpeakerArray('yBFC-RtfTfg', Speaker_to_ID))
arr_G7PydoX_WNQ = np.array(GetSpeakerArray('G7PydoX_WNQ', Speaker_to_ID))
arr_aMkNASF9lwE = np.array(GetSpeakerArray('aMkNASF9lwE', Speaker_to_ID))
arr_NAYkF04IZHI = np.array(GetSpeakerArray('NAYkF04IZHI', Speaker_to_ID))
arr_IBf9pXOmpFw = np.array(GetSpeakerArray('IBf9pXOmpFw', Speaker_to_ID))
arr_NuBtcUGqgMc = np.array(GetSpeakerArray('NuBtcUGqgMc', Speaker_to_ID))
arr_3Va3oY8pfSI = np.array(GetSpeakerArray('3Va3oY8pfSI', Speaker_to_ID))
arr_xqzLm0Xua8g = np.array(GetSpeakerArray('xqzLm0Xua8g', Speaker_to_ID))

print(arr_akiQuyhXR8o.shape)

InferencePredictions("SUCCESSIVE1_WRONG_akiQuyhXR8o", len(Successive1_F0), arr_akiQuyhXR8o, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry,Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('1')
InferencePredictions("SUCCESSIVE1_WRONG_XAcARiiK5uY", len(Successive1_F0), arr_XAcARiiK5uY, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('2')
InferencePredictions("SUCCESSIVE1_WRONG_qtcWebAYmKY", len(Successive1_F0), arr_qtcWebAYmKY, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('3')
InferencePredictions("SUCCESSIVE1_WRONG_0d6iSvF1UmA", len(Successive1_F0), arr_0d6iSvF1UmA, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('4')
InferencePredictions("SUCCESSIVE1_WRONG_x1qkuvUxuI", len(Successive1_F0), arr__x1qkuvUxuI, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('5')
InferencePredictions("SUCCESSIVE1_WRONG_NXfYNdapq3Q", len(Successive1_F0), arr_NXfYNdapq3Q, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('6')
InferencePredictions("SUCCESSIVE1_WRONG__MBiP3G2Pzc", len(Successive1_F0), arr__MBiP3G2Pzc, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('7')
InferencePredictions("SUCCESSIVE1_WRONG_xHHb7R3kx40", len(Successive1_F0), arr_xHHb7R3kx40, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('8')
InferencePredictions("SUCCESSIVE1_WRONG_apbSsILLh28", len(Successive1_F0), arr_apbSsILLh28, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('9')
InferencePredictions("SUCCESSIVE1_WRONG_yBFC_RtfTfg", len(Successive1_F0), arr_yBFC_RtfTfg, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('10')
InferencePredictions("SUCCESSIVE1_WRONG_G7PydoX_WNQ", len(Successive1_F0), arr_G7PydoX_WNQ, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('11')
InferencePredictions("SUCCESSIVE1_WRONG_aMkNASF9lwE", len(Successive1_F0), arr_aMkNASF9lwE, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('12')
InferencePredictions("SUCCESSIVE1_WRONG_NAYkF04IZHI", len(Successive1_F0), arr_NAYkF04IZHI, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('13')
InferencePredictions("SUCCESSIVE1_WRONG_IBf9pXOmpFw", len(Successive1_F0), arr_IBf9pXOmpFw, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('14')
InferencePredictions("SUCCESSIVE1_WRONG_NuBtcUGqgMc", len(Successive1_F0), arr_NuBtcUGqgMc, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('15')
InferencePredictions("SUCCESSIVE1_WRONG_3Va3oY8pfSI", len(Successive1_F0), arr_3Va3oY8pfSI, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('16')
InferencePredictions("SUCCESSIVE1_WRONG_xqzLm0Xua8g", len(Successive1_F0), arr_xqzLm0Xua8g, Successive1_F0, Successive1_bert, Successive1_AU1, Successive1_AU2, Successive1_AU4,Successive1_AU5, Successive1_AU6, Successive1_AU7, Successive1_Rx, Successive1_Ry, Successive1_Rz, ID_to_AU, ID_to_RX, ID_to_RY, ID_to_RZ)
print('17')
