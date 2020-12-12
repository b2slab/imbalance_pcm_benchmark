### from __future__ import division, absolute_import

from __future__ import division, absolute_import, print_function

import sys
import random

import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics 
import bisect
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils


#root
absPath = '/home/angela3/imbalance_pcm_benchmark/'
sys.path.insert(0, absPath)
absPath

#from src.Target import Target

np.random.seed(8)
random.seed(8)

#para no imprimir warnings chungos
import warnings 
warnings.simplefilter("ignore")


#from src.model_auxiliar_functions import *
#from src.training_auxiliar_functions import *
from src.Target import Target

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
        fingerprint = fingerprint.decode("utf-8") 
        fps = [float(n) for n in fingerprint]
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

def plot_history(path_history, path_to_save):
    """Plot evolution of accuracy and loss both in training and in validation sets"""
    with open(path_history, "rb") as input_file:
        history = pickle.load(input_file)
    history = history.history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    file_fig = ''.join((path_to_save, "/history_accuracy.png"))
    plt.savefig(file_fig)
    plt.clf()
    #plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    file_fig = ''.join((path_to_save, "/history_loss.png"))
    plt.savefig(file_fig)
    plt.clf()
    #plt.show()    
    return history

def load_best_model(history, path_to_cp):
    """Looks for the best epoch in terms of validation accuracy and load the corresponding model"""
    #which one is the best epoch?
    best_epoch = str(history['val_accuracy'].index(max(history['val_accuracy'])) +1).zfill(3)
    print('best epoch: ', best_epoch)
    cp_path = ''.join(string for string in [path_to_cp, '/weights-improvement-', best_epoch, '*.hdf5'])
    cp_path = glob(cp_path)[0]
    model = load_model(cp_path)
    return model, cp_path

def predict_on_test(input_list, y_test, model):
    """Load test data and predict on it"""
    #prediction
    y_predprob = model.predict(input_list)
    y_pred = y_predprob.argmax(axis=-1)
    y_prob = y_predprob[:,1]
    return y_predprob, y_pred, y_prob

def confusion_matrix(y_test_scalar, y_pred, path_to_confusion):
    """Creating a confusion matrix and saving it"""
    #model report
    print ("\nModel Report")
    print ("Accuracy (test set): %.4g" % metrics.accuracy_score(y_test_scalar, y_pred))
    print("Confusion matrix:")
    print (metrics.confusion_matrix(y_test_scalar, y_pred))
    print("Detailed classification report:")
    print (metrics.classification_report(y_test_scalar, y_pred))

    #Saving metrics 
    #file_out = ''.join(string for string in [absPath, 'data/checkpoint/',folder, '/', model_type, '/resulting_metrics.pickle'])
    file_out = ''.join((path_to_confusion, '/resulting_metrics.pickle'))
    d = (metrics.accuracy_score(y_test_scalar, y_pred), metrics.confusion_matrix(y_test_scalar, y_pred), 
     metrics.classification_report(y_test_scalar, y_pred, output_dict=True)) 

    with open(file_out, "wb") as output_file:
        pickle.dump(d, output_file)
        
def compute_roc(y_test_scalar, y_prob, path_to_auc):
    """Computing ROC curve and plotting it"""
    #Print model report:
    print ("\nModel Report II part")
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(y_test_scalar, y_prob))   

    #Saving metrics 
    #file_auc = ''.join(string for string in [absPath, 'data/checkpoint/', folder, '/', model_type, '/auc.pickle']) 
    file_roc = ''.join(string for string in [path_to_auc, '/roc.pickle'])
    with open(file_roc, "wb") as output_file:
        pickle.dump(metrics.roc_curve(y_test_scalar, y_prob), output_file)
        
    file_auc = ''.join(string for string in [path_to_auc, '/auc.pickle'])
    with open(file_auc, "wb") as output_file:
        pickle.dump(metrics.roc_auc_score(y_test_scalar, y_prob), output_file)
    
    # Computing ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test_scalar, y_prob)
    fig = plt.figure(figsize=(9,7))
    lw = 3
    plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title("ROC Curve w/ AUC=%s" % str(metrics.auc(fpr,tpr)), fontsize = 18)
    file_fig = ''.join(string for string in [path_to_auc, '/auc.png'])
    plt.savefig(file_fig)
    plt.clf()
    #plt.show()
    

#Functions to compute and plot AUC
def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    """Computing AUC from FPR and TPR"""
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in list(zip(ft[: -1], ft[1: ])):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    """Computing new FPR and TPR specified by threshold"""
    p = bisect.bisect_left(fpr, thresh)
    fpr = fpr.copy()
    fpr[p] = thresh
    return fpr[: p + 1], tpr[: p + 1]

def computing_partial_auc(y_test_scalar, y_prob, fold, hyperas_folder, thresh=0.05):
    #fpr, tpr, thresh, trapezoid=False):
    fpr, tpr, _ = metrics.roc_curve(y_test_scalar, y_prob)
    """Computing partial AUC at a given threshold"""
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, thresh)
    part_auc_notrapez = auc_from_fpr_tpr(fpr_thresh, tpr_thresh)
    part_auc_trapez = auc_from_fpr_tpr(fpr_thresh, tpr_thresh, True)
    print("Partial AUC:", part_auc_notrapez, part_auc_trapez)
    
    #Saving partial AUC
    #file_pauc = ''.join(string for string in [absPath, 'data/results/', folder, '/', model_type, '/pauc.pickle']) 
    file_pauc = ''.join(( absPath, "data/results/",  hyperas_folder, "/", str(fold), '/pauc.pickle'))

    with open(file_pauc, "wb") as output_file:
        pickle.dump(part_auc_notrapez, output_file)
    return part_auc_notrapez, part_auc_trapez
