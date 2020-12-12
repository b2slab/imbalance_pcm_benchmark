from __future__ import division, absolute_import, print_function

import sys
import random
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils

#root
absPath = '/home/angela3/imbalance_pcm_benchmark/'
sys.path.insert(0, absPath)
absPath

from src.Target import Target

np.random.seed(8)
random.seed(8)


def processing_sequences(batch_sequences, max_len, type_padding="post_padding"):
    """Processing amino acid sequences to be fed to the generator"""
    padding_short = type_padding.split("_")[0]
    instarget = Target('AAAAA')
    aa_to_int = instarget.predefining_dict()
    list_sequences = list(batch_sequences)
    list_targets = [Target(x) for x in list_sequences]
    #1st: we pad the sequences to the chosen max_len, with the strategy defined in type_padding
    list_padded = [x.padding_seq_position(max_len, padding_short) for x in list_targets]
    # 2nd: we convert amino acid sequences to integer sequences
    seqs_int = [instarget.string_to_int(x, aa_to_int) for x in list_padded]
    #this is simply to convert it to an array, I am NOT padding again
    seqs_int_array = sequence.pad_sequences(sequences=seqs_int, maxlen=max_len)
    seqs_onehot = instarget.int_to_onehot(list(seqs_int_array), len(aa_to_int))
    return seqs_onehot

def processing_fingerprints(batch_compounds):
    """Processing fingerprints to be fed to the generator"""
    list_compounds = list(batch_compounds)
    #nbits = len(list_compounds[0])
    lista_fps = []
    for fingerprint in list_compounds:
        fps = [int(n) for n in fingerprint]
        lista_fps.append(fps)
    comps_fps = sequence.pad_sequences(lista_fps)
    return comps_fps


def bin_to_onehot(y, num_classes=2):
    """From the label, creates a one hot label: 0.0 inactive, 1.0 active"""
    label_onehot = [np_utils.to_categorical(x, num_classes) for x in list(y)]
    return label_onehot

def batch_generator_DL(batch_size, f, group, table, indices, max_len_prot, type_padding_prot="post_padding"):
    """It generates batches for the deep learning model"""
    while True:
        sample_size = len(indices)
        #pre-defining dict
        #instarget = Target('AAAAAA')
        remnant = sample_size%batch_size       
        if remnant == 0:
            n_batches = int(sample_size/batch_size)
        else:
            n_batches = int(sample_size/batch_size) + 1
        #while True:
        for i in range(n_batches):
            if i == (n_batches-1):
                idcs = indices[i*batch_size:sample_size]
                #PROTEINS
                batch_sequences = list(f[group][table][idcs]["sequence"])
                #COMPOUNDS
                batch_compounds = list(f[group][table][idcs]["fingerprint"])
                #LABELS
                batch_y = list(f[group][table][idcs]["label"])
                
            else:
                idcs = indices[i*batch_size:i*batch_size+batch_size]
                #PROTEINS
                batch_sequences = list(f[group][table][idcs]["sequence"])
                #COMPOUNDS
                batch_compounds = list(f[group][table][idcs]["fingerprint"])
                #LABELS
                batch_y = list(f[group][table][idcs]["label"])
                #processing sequences and compounds
            seqs_onehot = np.asarray(processing_sequences(batch_sequences, max_len_prot, type_padding_prot))
            comps_batch = np.asarray(processing_fingerprints(batch_compounds))
            batch_labels = np.asarray(bin_to_onehot(batch_y))
            yield ([seqs_onehot, comps_batch], batch_labels)