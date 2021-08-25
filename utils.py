 #utils_seq2seq
import pandas as pd
import os
import pickle
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import tensorflow.keras.backend as K
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score)
from multiprocessing import Pool
from itertools import product
from functools import partial
import multiprocessing as mp
from multiprocessing import Process, Value, Array
from itertools import chain 
from tensorflow.keras.utils import Sequence
from sklearn.manifold import TSNE
import seaborn as sns
from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
from distinctipy import distinctipy
from scipy import stats
import time, re, os, io
from itertools import chain
import random
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error

path = os.getcwd()



def SaveResults_XLSX(results, Speaker, ID):
    results = np.array(results)
    with open('Speak_'+Speaker+'_AU'+ID+'.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(results)

def GetSpeakerArray(ID, Speaker_to_ID):
   arr = []
   for i in range(100):
      arr.append(ID)
   arrAll = []
   for i in range(10):
       arrAll.append(arr)
   arr=arrAll
   arrAll=[]
   for i in range(50):
       arrAll.append(arr)
   arr=arrAll
   arr=np.array(arr)
   arr = IntegerEncodeAU(Speaker_to_ID, arr)
   return arr


def CalculateMetrics(allpredictions, allgroundtruth, i, name):
    PREDS = allpredictions
    GNDTRUTH = allgroundtruth

    #Pearson Correlation - Ground Truth
    correlation_1_2, p_value = stats.pearsonr(PREDS, GNDTRUTH)
    RMSE_AU = sqrt(mean_squared_error(PREDS, GNDTRUTH))

    #classification : 0 for AU<0.5, 1 otherwise
    Threshold1 = 0.5
    Threshold2 = 2
    Threshold3 = 3
    Threshold4 = 4
    AURaw_T1 = classifyAU(GNDTRUTH, Threshold1)
    AUPred_T1 = classifyAU(PREDS, Threshold1)
    AURaw_T2 = classifyAU(GNDTRUTH, Threshold2)
    AUPred_T2 = classifyAU(PREDS, Threshold2)
    AURaw_T3 = classifyAU(GNDTRUTH, Threshold3)
    AUPred_T3 = classifyAU(PREDS, Threshold3)
    AURaw_T4 = classifyAU(GNDTRUTH, Threshold4)
    AUPred_T4 = classifyAU(PREDS, Threshold4)

    if name =="SUCCESSIVE1_RIGHT_":
        #plot activation
        PLOTActivation(i, AURaw_T1, AUPred_T1, Threshold1, name)
        PLOTActivation(i, AURaw_T2, AUPred_T2, Threshold2, name)
        PLOTActivation(i, AURaw_T3, AUPred_T3, Threshold3, name)
        PLOTActivation(i, AURaw_T4, AUPred_T4, Threshold4, name)

    AU_AHR = AHR(AURaw_T1, AUPred_T1)
    AU_NAHR = NAHR(AURaw_T1, AUPred_T1)
    AU_acc = accuracy(AURaw_T1, AUPred_T1)
    lines = ["PCC: "+str(correlation_1_2), "RMSE: "+str(RMSE_AU), "Accuracy: "+str(AU_acc), "AHR: "+ str(AU_AHR), "NAHR: "+str(AU_NAHR) ]

    if i == 1 :
        with open(name+"_AU1.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i == 2:
        with open(name+"_AU2.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i == 3 :
        with open(name+"_AU4.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i == 4:
        with open(name+"_AU5.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i == 5 :
        with open(name+"_AU6.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i == 6:
        with open(name+"_AU7.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i ==7:
        with open(name+"_RX.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i ==8:
        with open(name+"_RY.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    if i ==9:
        with open(name+"_RZ.txt", 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')


def PLOTActivation(i, GroundTruth, Prediction, threshold, name):

    time = list(range(0, len(GroundTruth)))
    threshold = str(threshold)

    #plot activation
    fig = plt.figure()
    plt.ylim([0,5])
    plt.xlim([0,31])
    plt.xlabel("Time [sec]")
    plt.ylabel("Intensity")
    plt.scatter(time, GroundTruth, s=160, c = 'blue', alpha=0.9, label="Ground truth")
    plt.scatter(time, Prediction, s=160, c = 'red',  alpha=0.6, label="Detected")
    plt.legend(loc="lower left")

    if i ==1:
        plt.title("Activation of AU01 - Threshold = "+ threshold)
        fig.savefig("AU01_Activation_"+threshold+"_"+name+".png")
    if i==2:
        plt.title("Activation of AU02 - Threshold = "+ threshold)
        fig.savefig("AU02_Activation_"+threshold+"_"+name+".png")
    if i ==3:
        plt.title("Activation of AU04 - Threshold = "+ threshold)
        fig.savefig("AU04_Activation_"+threshold+"_"+name+".png")
    if i==4:
        plt.title("Activation of AU05 - Threshold = "+ threshold)
        fig.savefig("AU05_Activation_"+threshold+"_"+name+".png")
    if i==5:
        plt.title("Activation of AU06 - Threshold = "+ threshold)
        fig.savefig("AU06_Activation_"+threshold+"_"+name+".png")
    if i==6:
        plt.title("Activation of AU07 - Threshold = "+ threshold)
        fig.savefig("AU07_Activation_"+threshold+"_"+name+".png")

    

def TurnIPU_MergedWords(matrix):
    to_return = []
    for i in matrix:
        word = i.flatten()
        word = list(word[~np.isnan(word)])
        word = list(filter((7.0).__ne__, word))
        word = list(filter((9.0).__ne__, word))
        word = list(filter((8.0).__ne__, word))
        word.insert(0, 8.0)
        word.append(9.0)
        if len(word)>100:
            word = word[:99]
        if len(word)<100:
            while len(word)<100:
                word.append(7.0)
        to_return.append(word)
    return np.array(to_return)

'''
def replace_with_dict_1(ar, dic):
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))
    sidx = k.argsort()
    toReturn  = v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
    last=[]
    for i in toReturn:
        flattened_list = list(chain.from_iterable(list(i)))
        last.append(flattened_list)
    return last
'''
def replace_with_dict_2(ar, dic):
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))
    sidx = k.argsort()
    toReturn  = v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
    return toReturn


def IntegerEncodeAU(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded

def IntegerEncodeF0(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded

def IntegerDecode(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded


def OneHot_Encode(integer_encoded, LENvocab):
    onehot_encoded = list()
    for value in integer_encoded:
           encodedWord = []
           for v in value:
               number = [0 for _ in range((LENvocab))]
               if v == 0:
                   encodedWord.append(number)
               elif v != 0:
                   number[(v)] = 1
                   encodedWord.append(number)
           onehot_encoded.append(encodedWord)
    return np.array(onehot_encoded)


def classifyAU(listAU, threshold):
    toreturn = []
    for i in listAU:
        if i <=threshold:
            a=0
            toreturn.append(a)
        else:
            #a=1
            a=i
            toreturn.append(a)
    return toreturn


def convert2DListto1DList( list ):
    result = []
    for element in list:
        for subElement in element:
            result.append( subElement )

    return result


def accuracy(AURaw, AUPred):
    count = 0
    totalFrames = len(AURaw)

    for i, raw in enumerate(AURaw):
        if (raw == AUPred[i]):
            count = count+1

    return (count/totalFrames)*100

def  AHR(AURaw, AUPred):
    on_predicted = sum(AUPred)
    on_raw = sum(AURaw)
    if (on_raw!=0):
        return (on_predicted/on_raw)*100
    else:
        print(' on raw is zero')

def  NAHR(AURaw, AUPred):
    off_predicted = (AUPred).count(0)
    off_raw = (AURaw).count(0)
    if off_raw != 0:
        return (off_predicted/off_raw)*100
    else:
        print(" off_raw is zero")


def FrontEndClipping(AURaw, AUPred):
    flag = 'F'
    FEC = 0
    on_raw = sum(AURaw)
    for i, raw in enumerate (AURaw):
        if flag == 'F' and raw ==1 and AUPred[i] ==0:
            FEC = FEC +1
        elif raw ==1 and AUPred[i] ==1:
            flag = 'T'
        elif  raw ==0 and AUPred[i] ==0:
            flag = 'F'
    if on_raw!=0:
        return (FEC/on_raw)*100
    else:
        print(" On_raw is zero")


def MidSpeechClipping(AURaw, AUPred):
    flag = 'F'
    MSC = 0
    on_raw = sum(AURaw)
    print('on_raw: ', on_raw)

    for i, raw in enumerate (AURaw):
        if flag =='T' and raw ==1 and AUPred[i]==0:
            MSC = MSC +1
        elif raw ==1 and AUPred[i]==1:
            flag ='T'
        else:
            flag ='F'
    if on_raw!=0:
        return (MSC/on_raw)*100
    else:
        print(" on_raw is zero")

def CarryOVER(AURaw, AUPred):
    flag = 'F'
    OVER = 0
    NS = (AURaw).count(0)
    for i, raw in enumerate (AURaw):
        if flag =='T' and raw ==0 and AUPred[i]==1:
            OVER = OVER +1
        elif raw ==1 and AUPred[i]==1:
            flag = 'T'
        else:
            flag = 'F'
    if NS!=0:
        return (OVER/NS)*100
    else:
        print("NS is zero")

def NoiseDetectedAsSpeech(AURaw, AUPred):
    flag = 'T'
    NDS = 0
    NS = (AURaw).count(0)

    for i, raw in enumerate (AURaw):
        if flag =='T' and raw ==0 and AUPred[i]==1:
            NDS = NDS +1
        elif raw ==0 and AUPred[i]==0:
            flag = 'T'
        else:
            flag = 'F'
    if NS!=0:
        return (NDS/NS)*100
    else:
        print("NS is zero")


def one_hot_decode_sequence(sequence):
    arr = []
    for i in range(100):
        encoded_integer = np.argmax(sequence[0, i, :])
        AU_value = ID_to_AU[encoded_integer]
        arr.append(AU_value)
    return arr

def formatDF_out(DFAllData_AU01_np):
    AU01_out = []
    for count, i in enumerate(DFAllData_AU01_np):
        new = []
        for ii in i:
            if ii != 7.0:
                new.append(ii)
        new.append(9.0)
        while len(new)<124:
            new.append(7.0)
        AU01_out.append(new)
    return AU01_out


def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles( np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



def strat_plot(m, b, epochs=150):
        x = np.arange(1, epochs+1, 1)
        y = 1/(1+np.exp(-(x*m+b)))
        y = 1-y
        fig_strat_plot = plt.figure()
        plt.subplots(figsize=(16,4))
        plt.plot(x, y)
        plt.grid()
        plt.title('Chosen sample schedule:')
        plt.xlabel('Epoch')
        plt.ylabel('Chance of teacher forcing')
        fig_strat_plot.savefig('strat_plot.png')


def padBERT(BERT):
        to_pad = [0] * 768
        RTN = []
        RTN = [i for i in BERT]

        if len(BERT)<10:
                Nb_to_add = 10 - len(BERT)
                values = []
                for i in range(0, Nb_to_add):
                        RTN.append(to_pad)
        if len(BERT)>10:
                RTN = RTN[:10]

        return RTN

def reshape(AU01_in, AU01_target):
        AU01_in = np.array(AU01_in)
        AU01_target = np.array(AU01_target)
        AU01_in = np.reshape(AU01_in, (AU01_in.shape[0], 10, AU01_in[0].shape[1], AU01_in[0].shape[2]))
        AU01_target = np.reshape(AU01_target, (AU01_target.shape[0],10, AU01_target[0].shape[1], AU01_target[0].shape[2]))
        return AU01_in, AU01_target

    
def reshape_WordLevel(AU01_in, AU01_target):
        AU01_in = np.array(AU01_in)
        AU01_target = np.array(AU01_target)
        AU01_in = np.reshape(AU01_in, (AU01_in.shape[0], AU01_in[0].shape[1], AU01_in[0].shape[2]))
        AU01_target = np.reshape(AU01_target, (AU01_target.shape[0], AU01_target[0].shape[1], AU01_target[0].shape[2]))
        return AU01_in, AU01_target

                                        
def OneHot_Encode(integer_encoded, LENvocab):
        onehot_encoded = []
        for value in integer_encoded:
                encodedWord = []
                for v in value:
                        number = [0 for _ in range((LENvocab))]
                        if v == 0:
                                encodedWord.append(number)
                        elif v != 0:
                                number[(v)] = 1
                                encodedWord.append(number)
                onehot_encoded.append(encodedWord)
        return np.array(onehot_encoded)


def cropOutputs(x):
        #x[0] is decoded at the end
        #x[1] is inputs
        #both have the same shape
        #padding = 1 for actual data in inputs, 0 for 0

        padding =  K.cast(K.not_equal(x[1],0), dtype=K.floatx())
        #if you have zeros for non-padded data, they will lose their backpropagation
        print('inside cropOutputs, X : ', x)
        print('inside cropOutputs, padding: ', padding)

        return x[0]*padding
                                        

def nospecial(text):
        import re
        text = re.sub("[^a-zA-Z0-9]+", "",text)
        return text

def load_model(model_filename, model_weights_filename):
        with open(model_filename, 'r', encoding='utf8') as f:
              model = tf.keras.models.model_from_json(f.read())
        model.load_weights(model_weights_filename)
        return model


def load_preprocess(path):	
	with open(path, mode='rb') as in_file:
		return pickle.load(in_file)

def getBERT_Timing_Words(BERT_Timing_Words_file):
        BERT_Timing_Words = []
        with open(BERT_Timing_Words_file, 'r') as rr:
                for line in rr:
                        line = line.strip()
                        line = line.replace('[', '')
                        line = line.replace(']', '')
                        line = line.replace("'", '')
                        elements = line.split(',')
                        count=0
                        #e = []
                        for i, element in enumerate(elements):
                                count = count+1
                                if(count<=128)&(len(element)>1):
                                        #e.append(element)
                                        BERT_Timing_Words.append(element)
        return BERT_Timing_Words

def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3         


def getIPUNumber(StartEndIPU, StartEndWord, filenameIPU, filenamesF0, i):

        IPUNumber = []
        filename_IPU = filenameIPU[i]
        startIPU = float(StartEndIPU[i][0])
        endIPU = float(StartEndIPU[i][1])

        for j, interval2 in enumerate(StartEndWord):
                        startWord = float(interval2.split()[0])
                        endWord = float(interval2.split()[1])
                        word = interval2.split()[2]
                        filename_word = filenamesF0[j]
                
                        if(filename_IPU == filename_word):
                                if(startIPU<= startWord and endWord<=endIPU):
                                        IPUNumber.append(i)
                                        
        return IPUNumber

def SaveCSV_TEDxDataBase(f):
        for filename in f:
                filename = filename.rstrip()
                print(filename)
                F0_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE_F0_Normalized/PreprocessedF0_"+ filename+".p"
                AU_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE__AU_0IntegerPadding/AUPreprocessed_"+ filename+".p"
                IPU_path= "/net/arpeggio/data2/anasynth_nonbp/fares/PICKLE_IPU_WithBreaks_Timing_new/IPUs_"+ filename+".p"
                
                
                isExistF0 = os.path.exists(F0_path)
                if isExistF0:
                        #load data
                        Vocab1_F0, Vocab2_F0, DataFrame_Freq, WORDS_F0 = load_preprocess(F0_path)
                        AU01, AU02, AU04, AU05, AU06, AU07, WORDS_AU = load_preprocess(AU_path)
                        IPU = load_preprocess(IPU_path)


                        DataFrame_Freq.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/F0/" + filename+".csv")
                        AU01.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU01_"+ filename+".csv")
                        AU02.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU02_"+ filename+".csv")
                        AU04.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU04_"+ filename+".csv")
                        AU05.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU05_"+ filename+".csv")
                        AU06.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU06_"+ filename+".csv")
                        AU07.to_csv(r"/net/bonsho/data2/anasynth_nonbp/fares/TEDxDatabase/ActionUnits/AU07_"+ filename+".csv")

                                                                 

def PreprocessData(F0_path, AU_path, IPU_path, BERT_path, filename):

    wordsPerIPU = []    
    #empty dataframes
    DFAllData_AU01 = pd.DataFrame()
    DFAllData_AU02 = pd.DataFrame()
    DFAllData_AU04 = pd.DataFrame()
    DFAllData_AU05 = pd.DataFrame()
    DFAllData_AU06 = pd.DataFrame()
    DFAllData_AU07 = pd.DataFrame()
    DFAllData_Tx = pd.DataFrame()
    DFAllData_Ty = pd.DataFrame()
    DFAllData_Tz = pd.DataFrame()
    DFAllData_Rx = pd.DataFrame()
    DFAllData_Ry = pd.DataFrame()
    DFAllData_Rz = pd.DataFrame()    
    DFAllData_F0 = pd.DataFrame()
    BERT_DF = pd.DataFrame()
    IPU_DF = pd.DataFrame()
    
    List_IPU = []
    List_BERT = []
    WORDS__F0 = []
    WORDS__AU = []
    Vocab1_F0 = {}
    Vocab2_F0 = {}
    BERT__Timing__Words = []

    count = 0
    
    #load data
    BERT__DF = load_preprocess(BERT_path)
    VOCAB, _, DataFrame_Freq, WORDS_F0 = load_preprocess(F0_path)
    AU01, AU02, AU04, AU05, AU06, AU07, Tx, Ty, Tz, Rx, Ry, Rz, WORDS_AU = load_preprocess(AU_path)
    IPU = load_preprocess(IPU_path)
                
    DataFrame_Freq.insert(0, "filename", filename)
    DataFrame_Freq.insert(1, "StartEnd", WORDS_F0)
    index = DataFrame_Freq.index
    DataFrame_Freq.reset_index()
    DataFrame_Freq.set_index(['filename', index], inplace = True)
    
    Rx.insert(0, "filename", filename)
    index = Rx.index
    Rx.reset_index()
    Rx.set_index(['filename', index], inplace = True)
    
    Ry.insert(0, "filename", filename)
    index = Ry.index
    Ry.reset_index()
    Ry.set_index(['filename', index], inplace = True)
    
    Rz.insert(0, "filename", filename)
    index = Rz.index
    Rz.reset_index()
    Rz.set_index(['filename', index], inplace = True)
    
    AU01.insert(0, "filename", filename)
    index = AU01.index
    AU01.reset_index()
    AU01.set_index(['filename', index], inplace = True)
    
    AU02.insert(0, "filename", filename)
    index = AU02.index
    AU02.reset_index()
    AU02.set_index(['filename', index], inplace = True)                                                
    
    AU04.insert(0, "filename", filename)
    index = AU04.index
    AU04.reset_index()
    AU04.set_index(['filename', index], inplace = True)
    
    AU05.insert(0, "filename", filename)
    index = AU05.index
    AU05.reset_index()
    AU05.set_index(['filename', index], inplace = True)
    
    AU06.insert(0, "filename", filename)
    index = AU06.index
    AU06.reset_index()
    AU06.set_index(['filename', index], inplace = True)
    
    AU07.insert(0, "filename", filename)
    index = AU07.index
    AU07.reset_index()
    AU07.set_index(['filename', index], inplace = True)                                                              

    #Stack the DataFrames on top of each other
    BERT_DF = BERT__DF
    DFAllData_F0 = DataFrame_Freq
    DFAllData_AU01 = AU01
    DFAllData_AU02 = AU02
    DFAllData_AU04 = AU04
    DFAllData_AU05 = AU05
    DFAllData_AU06 = AU06
    DFAllData_AU07 = AU07            
    DFAllData_Tx = Tx
    DFAllData_Ty = Ty
    DFAllData_Tz = Tz
    DFAllData_Rx = Rx
    DFAllData_Ry = Ry
    DFAllData_Rz = Rz
    indexIPUNumber = []
    
    IPUs = []
    for I in IPU:
        if I:
            start = (I[0][1])
            end = (I[len(I)-1][2])
            indexIPUNumber.append([start, end])
            words = []
            for j in I:
                words.append(j[0])
            IPUs.append(words)

    IPU__DF = pd.DataFrame(data = IPUs)
    IPU__DF.insert(0, "filename", filename)
    IPU__DF.insert(1, "StartEnd", indexIPUNumber)
    index = IPU__DF.index
    IPU__DF.reset_index()
    IPU__DF.set_index(["filename", index], inplace = True)
    IPU_DF = IPU__DF
    
    count = IPU_DF.shape[0]                
    WORDS__F0 = (WORDS_F0)
    WORDS__AU = (WORDS_AU)
    
    
    filenames = (DFAllData_F0.index.get_level_values(0).drop_duplicates(keep="first"))
    
    ########################################################## adapt F0 to AU ############################
    F0indexes = list(DFAllData_F0.index)
    AU01indexes = list(DFAllData_AU01.index)
    indexes_to_drop = []
    count = 0
    for i in range(0, DFAllData_AU01.shape[0]):
        if F0indexes[i+count] != AU01indexes[i]:
            indexes_to_drop.append(i)
            count = count + 1

    indexes_to_keep = set(range(DFAllData_F0.shape[0])) - set(indexes_to_drop)
    DFAllData_F0 = DFAllData_F0.take(list(indexes_to_keep))
    WORDS__F0 = [WORDS__F0[x] for x in indexes_to_keep]
    WORDS__AU = [WORDS__AU[x] for x in indexes_to_keep]

    for i, word in enumerate(WORDS_AU):
        start = word.split()[0]
        end = word.split()[1]
        word = word.split()[2]
        WORDS_AU[i] = start+' '+end+' '+word
            
    DFAllData_AU01.insert(1, "StartEnd", WORDS__AU)
    DFAllData_AU02.insert(1, "StartEnd", WORDS__AU)
    DFAllData_AU04.insert(1, "StartEnd", WORDS__AU)
    DFAllData_AU05.insert(1, "StartEnd", WORDS__AU)
    DFAllData_AU06.insert(1, "StartEnd", WORDS__AU)
    DFAllData_AU07.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Tx.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Ty.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Tz.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Rx.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Ry.insert(1, "StartEnd", WORDS__AU)
    DFAllData_Rz.insert(1, "StartEnd", WORDS__AU)
    
    indexAU01 = DFAllData_AU01.index
    indexAU02 = DFAllData_AU02.index
    indexAU04 = DFAllData_AU04.index
    indexAU05 = DFAllData_AU05.index
    indexAU06 = DFAllData_AU06.index
    indexAU07 = DFAllData_AU07.index
    indexTx = DFAllData_Tx.index
    indexTy = DFAllData_Ty.index
    indexTz = DFAllData_Tz.index
    indexRx = DFAllData_Rx.index
    indexRy = DFAllData_Ry.index
    indexRz = DFAllData_Rz.index
    
    indexF0 = DFAllData_F0.index  
    DFAllData_AU01.reset_index()
    DFAllData_AU02.reset_index()
    DFAllData_AU04.reset_index()
    DFAllData_AU05.reset_index()
    DFAllData_AU06.reset_index()
    DFAllData_AU07.reset_index()
    DFAllData_Tx.reset_index()
    DFAllData_Ty.reset_index()
    DFAllData_Tz.reset_index()
    DFAllData_Rx.reset_index()
    DFAllData_Ry.reset_index()
    DFAllData_Rz.reset_index()
    
    DFAllData_F0.set_index(["StartEnd", indexF0], inplace = True)
    DFAllData_AU01.set_index(["StartEnd", indexAU01], inplace = True)
    DFAllData_AU02.set_index(["StartEnd", indexAU02], inplace = True)
    DFAllData_AU04.set_index(["StartEnd", indexAU04], inplace = True)
    DFAllData_AU05.set_index(["StartEnd", indexAU05], inplace = True)
    DFAllData_AU06.set_index(["StartEnd", indexAU06], inplace = True)
    DFAllData_AU07.set_index(["StartEnd", indexAU07], inplace = True)
    DFAllData_Tx.set_index(["StartEnd", indexTx], inplace = True)
    DFAllData_Ty.set_index(["StartEnd", indexTy], inplace = True)
    DFAllData_Tz.set_index(["StartEnd", indexTz], inplace = True)
    DFAllData_Rx.set_index(["StartEnd", indexRx], inplace = True)
    DFAllData_Ry.set_index(["StartEnd", indexRy], inplace = True)
    DFAllData_Rz.set_index(["StartEnd", indexRz], inplace = True)
    
    DFAllData_F0.reset_index()
    DFAllData_AU01.reset_index()
    DFAllData_AU02.reset_index()
    DFAllData_AU04.reset_index()
    DFAllData_AU05.reset_index()
    DFAllData_AU06.reset_index()
    DFAllData_AU07.reset_index()
    DFAllData_Tx.reset_index()
    DFAllData_Ty.reset_index()
    DFAllData_Tz.reset_index()
    DFAllData_Rx.reset_index()
    DFAllData_Ry.reset_index()
    DFAllData_Rz.reset_index()
    
    
    filenamesF0 = (DFAllData_F0.index.get_level_values(1))
    StartEndWord = (DFAllData_F0.index.get_level_values(0))
    StartEndIPU = IPU_DF["StartEnd"]
    filenameIPU = IPU_DF.index.get_level_values(0)    
    
    #getting the IPU number of each word, so that we can add it as a column index to each F0/BERT/AU01
    IPUNumb = []    
    
    for i, interval in enumerate(StartEndIPU):
            filename_IPU = filenameIPU[i]
            startIPU = float(interval[0])
            endIPU = float(interval[1])

            for j, interval2 in enumerate(StartEndWord):
                    startWord = float(interval2.split()[0])
                    endWord = float(interval2.split()[1])
                    word = interval2.split()[2]
                    filename_word = filenamesF0[j]

                    if(filename_IPU == filename_word):
                            if(startIPU<= startWord and endWord<=endIPU):
                                    IPUNumb.append(i)
	
    
    return BERT_DF, IPUNumb, IPU_DF, DFAllData_AU01, DFAllData_AU02, DFAllData_AU04, DFAllData_AU05, DFAllData_AU06, DFAllData_AU07, DFAllData_Tx, DFAllData_Ty, DFAllData_Tz, DFAllData_Rx, DFAllData_Ry, DFAllData_Rz, DFAllData_F0, VOCAB


 
def buildVocab(quant):
        AU_to_ID = {}
        for i in range(1, len(quant)+3):

                if i == 1:
                        AU_to_ID[8] = i
                if i==2:
                        AU_to_ID[9] = i

                if i>2:
                        AU_to_ID[quant[i-3]] = i

        AU_to_ID[7.0] = 0
        ID_to_AU = {v:k for k, v in AU_to_ID.items()}
        return ID_to_AU, AU_to_ID


def F0Quantization(F0, quantF0):
    F0Quantized = []
    F0 = F0.flatten().tolist()
    for f in F0:
        if not math.isnan(f):
            f = quantF0[np.argmin(np.array([abs(f - qt) for qt in quantF0]))]
        F0Quantized.append(f)
    return F0Quantized                    
    

def buildVocabF0(quant):
    F0_to_ID = {}
    F0_to_ID[0.000] = 0
    for i in range(1, len(quant)):
        F0_to_ID[quant[i]] = i
    ID_to_F0 = {v:k for k, v in F0_to_ID.items()}
    F0_to_ID = {float("{0:.4f}".format(v)):k for k, v in ID_to_F0.items()}
    F0_to_ID = {v:k for k, v in ID_to_F0.items()}
    return ID_to_F0, F0_to_ID

                                        
def plot_metrics(history, path):

        print(history.history.keys())

        fig_accuracy = plt.figure()
        plt.plot(history.history['accuracy'], 'b', label='Train', linewidth='2')
        plt.plot(history.history['val_accuracy'], 'r', label = 'Validation', linewidth='2')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig_accuracy.savefig(path + '_accuracy.png')

        fig_loss = plt.figure()
        plt.plot(history.history['loss'], 'b', label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], 'r', label='Validation', linewidth=2)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig_loss.savefig(path + '_Loss.png')
                                                                                                                                                                                  
def plot_cm(raw, predictions, p=0.5):

        cm = confusion_matrix(raw, predictions)
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted" )
        fig.savefig('confusion matrix.png')


def plot_all_metrics(history):
 metrics = ["loss",
            "tp", "fp", "tn", "fn",
            "accuracy",
            "precision", "recall",
            "auc"]
 for n, metric in  enumerate(metrics):
         
         name = metric.replace("_", " ").capitalize()
         fig = plt.figure()
         plt.subplot(5, 2, n + 1)
         plt.plot(history.history.epoch, history.history[metric], color=colors[0], label="Train")
         plt.plot(history.history.epoch, history.history["val_"+metric], color=colors[1], linestyle="--", label="Val",)
         plt.xlabel("Epoch")
         plt.ylabel(name)
         if metric == "loss":
                 plt.ylim([0, plt.ylim()[1] * 1.2])
         elif metric == "accuracy":
                 plt.ylim([0.4, 1])
         elif metric == "fn":
                 plt.ylim([0, plt.ylim()[1]])
         elif metric == "fp":
                 plt.ylim([0, plt.ylim()[1]])
         elif metric == "tn":
                 plt.ylim([0, plt.ylim()[1]])
         elif metric == "tp":
                 plt.ylim([0, plt.ylim()[1]])
         elif metric == "precision":
                 plt.ylim([0, 1])
         elif metric == "recall":
                 plt.ylim([0.4, 1])
         else:
                 plt.ylim([0, 1])

         plt.legend()
         plt.savefig(metric+".png")



def Plot_Losses(history):
    
    print(history.keys())
    
    # Plot of all losses
    fig_loss = plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_.png')

    
    #plot AU01 loss
    fig_loss = plt.figure()
    plt.plot(history['AU1_loss'])
    plt.plot(history['val_AU1_loss'])
    plt.title('AU01 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU01.png')
    
    #plot AU02 loss
    fig_loss = plt.figure()
    plt.plot(history['AU2_loss'])
    plt.plot(history['val_AU2_loss'])
    plt.title('AU02 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU02.png')
    
    #plot AU04 loss
    fig_loss = plt.figure()
    plt.plot(history['AU4_loss'])
    plt.plot(history['val_AU4_loss'])
    plt.title('AU04 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU04.png')
    
    #plot AU05 loss
    fig_loss = plt.figure()
    plt.plot(history['AU5_loss'])
    plt.plot(history['val_AU5_loss'])
    plt.title('AU05 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU05.png')
    
    #plot AU06 loss
    fig_loss = plt.figure()
    plt.plot(history['AU6_loss'])
    plt.plot(history['val_AU6_loss'])
    plt.title('AU06 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU06.png')
    
    #plot AU07 loss
    fig_loss = plt.figure()
    plt.plot(history['AU7_loss'])
    plt.plot(history['val_AU7_loss'])
    plt.title('AU07 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_AU07.png')
    

    #plot Rx loss
    fig_loss = plt.figure()
    plt.plot(history['Rx_loss'])
    plt.plot(history['val_Rx_loss'])
    plt.title('Rx loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_Rx.png')
    
    #plot Ry loss
    fig_loss = plt.figure()
    plt.plot(history['Ry_loss'])
    plt.plot(history['val_Ry_loss'])
    plt.title('Ry loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_Ry.png')
    

    #plot Rz loss
    fig_loss = plt.figure()
    plt.plot(history['Rz_loss'])
    plt.plot(history['val_Rz_loss'])
    plt.title('Rz loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_loss.savefig('loss_Rz.png')
    
    
def Plot_TSNE(SPKEmbeddings, SPK):
    
    # t-SNE
    X = np.array(SPKEmbeddings)
    X = np.reshape(X, (X.shape[0], X.shape[2]))
    X = TSNE(n_components=2).fit_transform(X)
    
    # plot t-SNE clusters
    color_class = SPK
    speakers = set(color_class)
    print('length of color class is : ', set(color_class))
    
    # generate N visually distinct colours
    N = len(set(color_class))
    colors = distinctipy.get_colors(N)
    palette = sns.color_palette(colors)
    
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(30,30)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 11
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE111.png")
    
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(20,20)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 11
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE11.png")

    fig = plt.figure()
    sns.set(rc={'figure.figsize':(20,20)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE14.png")

    fig = plt.figure()
    sns.set(rc={'figure.figsize':(10,10)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 20
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE15.png")
    
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(30,30)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 12
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE16.png")
    
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(40,40)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 12
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE16.png")


    fig = plt.figure()
    sns.set(rc={'figure.figsize':(7,7)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("TSNE17.png")

def Plot_PCA(SPKEmbeddings, SPK):

    color_class = SPK
    speakers = set(color_class)
    print('length of color class is : ', set(color_class))

    # generate N visually distinct colours
    N = len(set(color_class))
    colors = distinctipy.get_colors(N)
    palette = sns.color_palette(colors)

                
            
    # PCA
    pca = PCA(n_components=2)
    X = np.array(SPKEmbeddings)
    X = np.reshape(X, (X.shape[0], X.shape[2]))
    pca.fit(X)
    print('pca.explained_variance_', pca.explained_variance_)

    """
These vectors represent the principal axes of the data, and the length of the vector is an indication of how "important" that axis is in describing the distribution of the data—more precisely, it is a measure of the variance of the data when projected onto that axis. The projection of each data point onto the principal axes are the "principal components" of the data.
"""
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)


    """
The transformed data has been reduced to a single dimension. To understand the effect of this dimensionality reduction, we can perform the inverse transform of this reduced data and plot it along with the original data
"""
    fontsize = 12
    fig = plt.figure()
    X_new = pca.inverse_transform(X_pca)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal');
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    fig.savefig("pca_inverse_transform.png")

    """
The light points are the original data, while the dark points are the projected version. This makes clear what a PCA dimensionality reduction means: the information along the least important principal axis or axes is removed, leaving only the component(s) of the data with the highest variance. The fraction of variance that is cut out (proportional to the spread of points about the line formed in this figure) is roughly a measure of how much "information" is discarded in this reduction of dimensionality.
"""
    # PCA for visualization
    pca = PCA(2)  # project from 64 to 2 dimensions
    X = np.array(SPKEmbeddings)
    X = np.reshape(X, (X.shape[0], X.shape[2]))
    projected = pca.fit_transform(X)
    print('digits.data.shape: ', X.shape)
    print('projected.shape: ', projected.shape)


    # plot the first two principal components of each point to learn about the data
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(30,30)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    fig.savefig("PCA_visualization.png")

    fig = plt.figure()
    sns.set(rc={'figure.figsize':(20,20)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    fig.savefig("PCA_visualization1.png")

    fig = plt.figure()
    sns.set(rc={'figure.figsize':(15,15)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    fig.savefig("PCA_visualization1.png")

    fig = plt.figure()
    sns.set(rc={'figure.figsize':(10,10)})
    sns_plot = sns.scatterplot(X[:,0], X[:,1], hue=color_class, palette=palette)
    fontsize = 14
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    fig.savefig("PCA_visualization2.png")


    
