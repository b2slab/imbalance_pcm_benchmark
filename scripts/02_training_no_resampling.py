from __future__ import division, absolute_import, print_function

import sys
import os
import pickle
import numpy as np
import random

import h5py
from sklearn import metrics 
from collections import Counter
from glob import glob
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from keras import backend as K 
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.backend import manual_variable_initialization 
from keras.models import load_model, Model
from keras.layers import Dense, concatenate, Flatten, Conv1D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, TerminateOnNaN
#import keras

#root
absPath = '/home/angela3/imbalance_pcm/'
sys.path.insert(0, absPath)


from src.model_auxiliar_functions import *
from src.Target import Target
from src.postproc_auxiliar_functions import *

os.environ['PYTHONHASHSEED'] = '0' 
np.random.seed(8)
random.seed(8)
tf.random.set_seed(8)


batch_size = 128
epochss = 100
type_padding_prot = "post_padding"


#Opening HDF5 with data
file_h5 = "".join((absPath, 'data/data_no_resampled/compounds_activity.h5'))
f = h5py.File(file_h5, 'r')
group = '/activity'
table = "prot_comp"

#Loading maximum lengths of proteins and compounds
with open("".join((absPath, 'data/prot_max_len.pickle')), "rb") as input_file:
    max_len_prot = pickle.load(input_file)
#Defining protein dictionary    
instarget = Target("AAA")
prot_dict = instarget.predefining_dict()

learning_rate = 5e-4
decay_rate = learning_rate/epochss
adamm = Adam(lr=learning_rate, beta_1=0.1, beta_2=0.001, epsilon=1e-08, decay=decay_rate)

#Defining model
# LEFT BLOCK (to analyse amino acid sequences)
input_seq = Input(shape=(max_len_prot, len(prot_dict)), dtype='float32')
conv_seq = Conv1D(filters=64, padding='same', strides=1, kernel_size=3, activation='relu')(input_seq)
dropout_1 = Dropout(0.4)(conv_seq)
flatten_seq = Flatten()(dropout_1)#(dense_seq)
dense_seq_2 = Dense(50)(flatten_seq)
dropout_2 = Dropout(0.4)(dense_seq_2)

#RIGHT BRANCH (to analyse fingerprints)
input_fps = Input(shape=(881,), dtype='float32')
dense_fps = Dense(50)(input_fps)
dropout_3 = Dropout(0.4)(dense_fps)
#bn_3 =  BatchNormalization()(dense_fps)#(dense_seq_2)#(conv_seq)


#MERGE BOTH BRANCHES
main_merged = concatenate([dropout_2, dropout_3],axis=1)#([dense_seq_2, dense_fps], axis=1)

main_dense = Dense(2, activation='softmax')(main_merged)

#build and compile model
model = Model(inputs=[input_seq, input_fps], outputs=[main_dense])
model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])

model.summary()

for fold in range(nfolds):
    print("Starting fold: ", str(fold)) 
    
    file_list = "".join((absPath, "data/data_no_resampled/splitting_lists/splitting_", str(fold),"_list.pickle"))
    with open(file_list, "rb") as input_file:
        splitting_list = pickle.load(input_file)    
    
    splitting_list[0].sort()
    splitting_list[1].sort()
    #Defining generators
    train_generator = batch_generator_DL(batch_size, f, group, table, splitting_list[0], 
                                     max_len_prot, type_padding_prot=type_padding_prot)
    val_generator = batch_generator_DL(batch_size, f, group, table, splitting_list[1], 
                                     max_len_prot, type_padding_prot=type_padding_prot)

    
    #defining callbacks
    if not os.path.exists("".join((absPath, "data/data_no_resampled/logs/", str(fold), "/"))):
        os.makedirs("".join((absPath, "data/data_no_resampled/logs/", str(fold), "/")))
    
    log_path = "".join((absPath, "data/data_no_resampled/logs/", str(fold), "/training_log.csv"))
    csv_logger = CSVLogger(log_path)

    if not os.path.exists("".join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/"))):
        os.makedirs("".join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/")))

    #if there are already files in the folder, it removes them
    r = glob("".join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/*")))
    for i in r:
        os.remove(i)
   
    terminan = TerminateOnNaN()
    checkpoint_path = "".join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/weights-improvement-{epoch:03d}-{val_accuracy:.4f}.hdf5"))
    mcheckpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=0, 
                                          save_best_only=True, save_weights_only=False)
    #early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20, restore_best_weights=True)

    callbacks_list = [csv_logger, terminan, mcheckpoint ]
    
    # fitting the model
    history = model.fit_generator(generator=train_generator, 
                              validation_data=val_generator,
                             steps_per_epoch= int(len(splitting_list[0])/batch_size),
                              validation_steps=int(len(splitting_list[1])/batch_size),
                             epochs=epochss,
                             callbacks=callbacks_list,
                             verbose=1)
    #saving history
    if not os.path.exists("".join((absPath, "data/data_no_resampled/results/", str(fold), "/"))):
        os.makedirs("".join((absPath, "data/data_no_resampled/results/", str(fold), "/")))

    with open("".join((absPath, "data/data_no_resampled/results/", str(fold), "/history.pickle")), 'wb') as handle:
        pickle.dump(history, handle)
        
    print("History saved succesfully")
    
    #Prediction on test data
    splitting_list[2].sort()
    #PROTEINS
    batch_sequences = list(f[group][table][splitting_list[2]]["sequence"])
    #COMPOUNDS
    batch_compounds = list(f[group][table][splitting_list[2]]["fingerprint"])
    #LABELS
    batch_y = list(f[group][table][splitting_list[2]]["label"])
    #processing sequences and compounds
    seqs_onehot = np.asarray(processing_sequences(batch_sequences, max_len_prot, type_padding_prot))
    comps_batch = np.asarray(processing_fingerprints(batch_compounds))
    batch_labels = np.asarray(bin_to_onehot(batch_y))
    
    history_path = "".join((absPath, "data/data_no_resampled/results/", str(fold), "/history.pickle"))
    path_to_confusion = "".join((absPath, "data/data_no_resampled/results/", str(fold), "/"))
    path_to_auc = "".join((absPath, "data/data_no_resampled/results/", str(fold), "/"))
    
    history = plot_history(history_path, "".join((absPath, "data/data_no_resampled/results/", str(fold), "/")))
    path_to_cp = ''.join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/"))

    model, best_path = load_best_model(history, path_to_cp)

    cps_loc = ''.join((absPath, "data/data_no_resampled/checkpoint/", str(fold), "/*.hdf5")) 

    #removing the rest of weights
    fileList = glob(cps_loc, recursive=True)
    fileList.remove(best_path)
    if len(fileList) >1:
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")
    
    print("Starting prediction on test data")
    
    #y_predprob, y_pred, y_prob = predict_on_test(test_generator, y_test, model, splitting_list[2], batch_size)
    y_predprob = model.predict([seqs_onehot, comps_batch])
    y_prob = y_predprob[:,1]
    y_pred = y_predprob.argmax(-1)
    y_test = batch_labels.argmax(-1)
    print(Counter(y_pred))
    
    batch_compID_test = list(f[group][table][splitting_list[2]]["da_comp_id"])
    batch_protID_test = list(f[group][table][splitting_list[2]]["da_prot_id"])
    
    #confusion matrix
    confusion_matrix(y_test, y_pred, path_to_confusion)
        
    #AUC
    file_auc = ''.join((absPath, "data/data_no_resampled/results/", str(fold), "/AUC.pickle"))
    compute_roc(y_test, y_prob, path_to_auc)
    
    # saving predictions on test set

    predictions_test = pd.DataFrame({"y_test":y_test, "y_prob":y_prob, "y_pred":y_pred, "comp_ID": batch_compID_test,
                                "DeepAffinity Protein ID": batch_protID_test})

    if not os.path.exists("".join((absPath, "data/data_no_resampled/predictions/", str(fold), "/"))):
        os.makedirs("".join((absPath, "data/data_no_resampled/predictions/", str(fold), "/")))

    predictions_test.to_csv("".join((absPath, "data/data_no_resampled/predictions/", str(fold), "/test.csv")))
    
    print("Starting prediction on validation data")
    
    val_generator = batch_generator_DL(batch_size, f, group, table, splitting_list[1], 
                                     max_len_prot, type_padding_prot=type_padding_prot)

    y_predprob_val =  model.predict_generator(val_generator, steps= int(np.ceil(len(splitting_list[1])/batch_size)))

    y_pred_val = y_predprob_val.argmax(axis=-1)
    y_prob_val = y_predprob_val[:,1]
    
    print(Counter(y_pred_val))
    
    batch_compID_val = list(f[group][table][splitting_list[1]]["da_comp_id"])
    batch_protID_val = list(f[group][table][splitting_list[1]]["da_prot_id"])
    
    batch_y_val = list(f[group][table][splitting_list[1]]["label"])
    val_labels = np.asarray(bin_to_onehot(batch_y_val))
    y_val = val_labels.argmax(-1)
    
    val_test = pd.DataFrame({"y_val":y_val, "y_prob":y_prob_val, "y_pred":y_pred_val,
                         "comp_ID": batch_compID_val,
                                "DeepAffinity Protein ID": batch_protID_val})
    val_test.to_csv("".join((absPath, "data/data_no_resampled/predictions/", str(fold), "/validation.csv")))
    
    print("Starting prediction on training data")
    
    train_generator = batch_generator_DL(batch_size, f, group, table, splitting_list[0], 
                                     max_len_prot, type_padding_prot=type_padding_prot)
    y_predprob_train =  model.predict_generator(train_generator, steps= int(np.ceil(len(splitting_list[0])/batch_size)))

    y_pred_train = y_predprob_train.argmax(axis=-1)
    y_prob_train = y_predprob_train[:,1]
    
    print(Counter(y_pred_train))
    
    batch_compID_train = list(f[group][table][splitting_list[0]]["da_comp_id"])
    batch_protID_train = list(f[group][table][splitting_list[0]]["da_prot_id"])
    
    batch_y_train = list(f[group][table][splitting_list[0]]["label"])
    train_labels = np.asarray(bin_to_onehot(batch_y_train))
    y_train = train_labels.argmax(-1)
    
    train_df = pd.DataFrame({"y_train":y_train, "y_prob":y_prob_train, "y_pred":y_pred_train, 
                         "comp_ID": batch_compID_train,
                                "DeepAffinity Protein ID": batch_protID_train})
    train_df.to_csv("".join((absPath, "data/data_no_resampled/predictions/", str(fold), "/training.csv")))
    