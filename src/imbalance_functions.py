from __future__ import division, absolute_import

import sys
import os
import numpy as np
import pandas as pd
import random
import pickle
import h5py
import time
from sklearn import metrics

#root
absPath = '/home/angela3/imbalance_pcm_benchmark/'
sys.path.insert(0, absPath)

np.random.seed(8)
random.seed(8)

def prot_comp_composite(row):
    """Creates a composite column from the protein and compound IDs"""
    composi = row["DeepAffinity Compound ID"] + "," + row["DeepAffinity Protein ID"]
    return composi

def separating_fps(string):
    """ Separating fingerprints into numbers"""
    fps_separated = [int(char) for char in string]
    return fps_separated


def batch_generator(batch_size, df, sample_indexes):
    """It generates batches for the compounds K-means clustering"""
    #while True:
    remnant = len(sample_indexes)%batch_size       
    if remnant == 0:
        n_batches = int(len(sample_indexes)/batch_size)
    else:
        n_batches = int(len(sample_indexes)/batch_size) + 1
        
    for i in range(0, n_batches):    
        if i == (n_batches-1):
            batch_idcs = sample_indexes[i*batch_size:len(sample_indexes)]
        else:
            batch_idcs = sample_indexes[i*batch_size:i*batch_size+batch_size]
            
        lista_dicts = []        
        #print(batch_idcs)
        for j in batch_idcs:
            #print(j)
            dict_comp = df.loc[j].to_dict()
            #print(dict_comp)
            dict_comp.pop("fingerprint", None)
            lista_dicts.append(dict_comp)   
        yield(lista_dicts)        

def computing_active_inactive_ratio(df,prot_id):
    """Computing the ratio of actives respect to the total number of interactions per protein"""
    counting = df.loc[:,["DeepAffinity Protein ID", "label"]].groupby(["DeepAffinity Protein ID", "label"]).size()
    count_df = counting.to_frame(name = 'size').reset_index()
    count_prot = count_df.loc[count_df["DeepAffinity Protein ID"]==prot_id, :]
    if count_prot.shape[0] == 0:
        return np.nan
    n_labels = count_prot.shape[0]
    if n_labels == 1:
        label_label = count_prot["label"].values[0]
        if label_label == 1.0:
            ratio = 1.0
        else:
            ratio = 0.0
    else:
        n_actives = count_prot.loc[count_prot["label"]==1.0, "size"].values[0]
        n_total = n_actives + count_prot.loc[count_prot["label"]==0.0, "size"].values[0]
        ratio = n_actives/n_total
    return ratio
        
def accumulated_size_clusters(df):
    """From clusters classification performed by K-means, we compute the size of each cluster
    (both total and accumulative)"""
    #Counting how many compounds are for each cluster
    label_count = df['cluster_label'].value_counts()
    #Permuting clusters
    permutated_clusters = np.random.permutation(np.arange(0, max(df['cluster_label']+1), 1))
    clusters_size = [label_count[x] for x in permutated_clusters]
    clusters_accumulated_size = [sum(clusters_size[0:i+1]) for i in range(len(clusters_size))]
    clusters_acc_percent = [i/max(clusters_accumulated_size)*100 for i in clusters_accumulated_size]
    compounds_classif = pd.DataFrame({'cluster_label':permutated_clusters, 
                                      'size': clusters_size, 'accumulated_size': clusters_accumulated_size, 
                                      'clusters_acc_percent': clusters_acc_percent})
    return compounds_classif
    
def training_test_split(df, training_size, test_size, val_size, idx):
    """From the accumulated sizes of the clusters, we divide data between train, test and validation"""
    # Dividing in training, test and validation 
    conditions = [ df['clusters_acc_percent'] < training_size,
    (df['clusters_acc_percent'] > training_size) & (df['clusters_acc_percent'] <  training_size + test_size)]
    choices = [0,1]
    name_column = "splitting_" + str(idx)
    df[name_column] = np.select(conditions, choices, default=2)
    return df

def splitting_division(f, group, table, sample_indices, splitting_fold):
    """Saving indices of each splitting group to a list that will be fed later to the deep learning model"""
    t0 = time.time()
    #indices_train = [i for i in sample_indices if f[group][table][sample_indices[i]][splitting_fold] == 0.0]
    indices_train = [i for i in sample_indices if f[group][table][i][splitting_fold] == 0.0]
    #indices_val = [i for i in sample_indices if f[group][table][sample_indices[i]][splitting_fold] == 1.0]
    indices_val = [i for i in sample_indices if f[group][table][i][splitting_fold] == 1.0]
    #indices_test = [i for i in sample_indices if f[group][table][sample_indices[i]][splitting_fold] == 2.0]
    indices_test = [i for i in sample_indices if f[group][table][i][splitting_fold] == 2.0]
    print(time.time() - t0)
    return indices_train, indices_val, indices_test

    
def training_test_split_semi(df, training_size, test_size, val_size, idx):
    """From the accumulated sizes of the clusters, we divide data between train and validation for the semi_resampling strategy"""
    # Dividing in training, test and validation 
    conditions = [ df['clusters_acc_percent'] < training_size,df['clusters_acc_percent'] > training_size]
    choices = [0,1]
    name_column = "splitting_" + str(idx)
    df[name_column] = np.select(conditions,choices, default=0)
    return df

def splitting_division_semi(f, group, table, sample_indices, splitting_fold):
    """Saving indices of each splitting group to a list that will be fed later to the deep learning model.
    Specific for the semi_resampling strategy, since there only training and validation need to be splitted."""
    t0 = time.time()
    indices_train = [i for i in sample_indices if f[group][table][i][splitting_fold] == 0.0]
    indices_val = [i for i in sample_indices if f[group][table][i][splitting_fold] == 1.0]
    print(time.time() - t0)
    return indices_train, indices_val

def len_seq(row):
    """Computing the length of a protein sequence in a datframe"""
    leen = len(row["Sequence"])
    return leen

def converting_ratios_to_df(train_ratios, test_ratios, pred_ratios, metrics_list, strategy, prot_df):
    """Creating a dataframe from the results (ratio of actives from the training set, from the test set,
    from the predicted test set)"""
    ratios_training_df = pd.DataFrame(train_ratios)
    print(ratios_training_df.info())
    ratios_test_df = pd.DataFrame(test_ratios)
    print(ratios_test_df.info())
    ratios_predicted_test_df =  pd.DataFrame(pred_ratios)
    print(ratios_predicted_test_df.info())
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.info())
    m1 = pd.merge(ratios_training_df, ratios_test_df, "inner", on="DeepAffinity Protein ID")
    print(m1.info())
    m2 = pd.merge(m1, ratios_predicted_test_df, "inner", on="DeepAffinity Protein ID")
    print(m2.info())
    ratios_df = pd.merge(m2, metrics_df, "inner", on="DeepAffinity Protein ID")
    print(ratios_df.info())
    ratios_df["strategy"] = strategy
    print(ratios_df.info())
    print(ratios_df.head())
    ratios_seq = pd.merge(ratios_df, prot_df, "left", on="DeepAffinity Protein ID")
    return ratios_df

#loading predicted data in order to check ratio of predicted in test
def predictions_ratios_list(fold, pred_test, unique_prots):
    """Computing the ratio of actives respect to the total number of interactions in the predicted data"""
    only_pred = pred_test.loc[:, ["y_pred", "DeepAffinity Protein ID"]]
    only_pred.rename(columns = {"y_pred":"label"}, inplace=True)
    print(only_pred.info())
    
    #por alguna razon que desconozco en este caso no eran bytes, sino q la b' ' estaba en string
    only_pred["DeepAffinity Protein ID"] = only_pred["DeepAffinity Protein ID"].str.replace("b\'", "")
    only_pred["DeepAffinity Protein ID"] = only_pred["DeepAffinity Protein ID"].str.replace("\'", "")
    
    ratios_pred_list = []
    for prot in unique_prots:
        ratios_pred_test = {}
        ratios_pred_test["DeepAffinity Protein ID"] = prot
        ratios_pred_test["ratio_test_predicted"] = computing_active_inactive_ratio(only_pred, prot)
        ratios_pred_list.append(ratios_pred_test)
    
    pred_test["DeepAffinity Protein ID"] = pred_test["DeepAffinity Protein ID"].str.replace("b\'", "")
    pred_test["DeepAffinity Protein ID"] = pred_test["DeepAffinity Protein ID"].str.replace("\'", "")
    
    return ratios_pred_list, pred_test

def computing_metrics_per_prot(prot, pred_test):
    """For each protein, we compute the defined performance metrics and return all the values in a dictionary"""
    dict_prot = {}
    predictions_test_sub = pred_test[pred_test["DeepAffinity Protein ID"] == prot]
    if predictions_test_sub.shape[0] == 0:
        return None
    dict_prot["acc"] = metrics.accuracy_score(y_true=predictions_test_sub["y_test"].values, 
                                 y_pred=predictions_test_sub["y_pred"].values)
    try: 
        dict_prot["auroc"] = metrics.roc_auc_score(y_true = predictions_test_sub["y_test"].values, 
                                               y_score = predictions_test_sub["y_prob"].values)
    except ValueError:
        print("Only one class present in y_true.")
        dict_prot["auroc"] = np.nan
    dict_prot["f1"] = metrics.f1_score(y_true=predictions_test_sub["y_test"].values, 
                                       y_pred=predictions_test_sub["y_pred"].values)
    dict_prot["DeepAffinity Protein ID"] = prot
    dict_prot["balanced_acc"] = metrics.balanced_accuracy_score(y_true=predictions_test_sub["y_test"].values, 
                                                        y_pred=predictions_test_sub["y_pred"].values)
    dict_prot["mcc"] = metrics.matthews_corrcoef(y_true=predictions_test_sub["y_test"].values, 
                                         y_pred=predictions_test_sub["y_pred"].values)
    return dict_prot

def creating_ratios_list(training_set, fold, f, group, table, sample_indices, unique_prots):
    """Computing proportion of actives respect to the total number of interactions in a list."""
    prots = list(f[group][table][sample_indices]["da_prot_id"])
    labels = list(f[group][table][sample_indices]["label"])
    data_df = pd.DataFrame({"DeepAffinity Protein ID":prots, "label":labels})
    data_df["DeepAffinity Protein ID"] = data_df["DeepAffinity Protein ID"].str.decode("utf-8")
    print(data_df.info())
    ratios_list = []
    for prot in unique_prots:
        ratioss = {}
        column_name = "ratio_" + training_set
        ratioss["DeepAffinity Protein ID"] = prot
        ratioss[column_name] = computing_active_inactive_ratio(data_df, prot)
        ratios_list.append(ratioss)
    return ratios_list

def computing_random_baseline(ratios_df_completo, strategy, prot, fold, protein_type):
    """Computing the random baseline to compare our results with for each strategy and each fold. It is computed from the
    actives/inactives ratio in training set in each case."""
    subdf = ratios_df_completo[(ratios_df_completo.strategy==strategy) & (ratios_df_completo.fold == fold)]
    if subdf[subdf["DeepAffinity Protein ID"]==prot].shape[0] == 0:
        return None, None
    ratio_training = subdf.loc[subdf["DeepAffinity Protein ID"]==prot, "ratio_training"].dropna().values.mean()
    if np.isnan(ratio_training):
        return None, None
    pred_test_path = "".join((absPath, "data/", protein_type, "/", strategy, "/predictions/", str(fold), "/test.csv"))
    predtest_loaded = pd.read_csv(pred_test_path, index_col=False)
    predtest_loaded["DeepAffinity Protein ID"] = predtest_loaded["DeepAffinity Protein ID"].str.replace("b\'", "")
    predtest_loaded["DeepAffinity Protein ID"] = predtest_loaded["DeepAffinity Protein ID"].str.replace("\'", "")
    n_interactions = predtest_loaded[predtest_loaded["DeepAffinity Protein ID"]==prot].shape[0]
    #print(n_interactions)
    #n_interactions = int(subdf.loc[subdf["DeepAffinity Protein ID"]==prot, "n_interactions"].dropna().values.mean())
    #comps = predtest_loaded.loc[predtest_loaded["DeepAffinity Protein ID"]==prot, "comp_ID"].values.tolist()
    n_actives = int(round(ratio_training*n_interactions, 0))
    #print(n_actives)
    n_inactives = n_interactions-n_actives  
    #print(n_inactives)
    active_rdm_comps = [random.uniform(0.5, 1) for r in range(n_actives)]
    inactive_rdm_comps = [random.uniform(0, 0.49) for r in range(n_inactives)]
    rdm_comps = active_rdm_comps + inactive_rdm_comps
    random.shuffle(rdm_comps) 
    return rdm_comps#, comps

def predictions_ratios_list_bsl(strategy, pred_test, unique_prots, fold):
    """Computing the proportion of actives respect to the total number of interactions for the random_baseline"""
    only_pred = pred_test.loc[pred_test.strategy==strategy, ["label", "DeepAffinity Protein ID"]]
    print(only_pred.info())
    
    ratios_pred_list = []
    for prot in unique_prots:
        ratios_pred_test = {}
        ratios_pred_test["DeepAffinity Protein ID"] = prot
        ratios_pred_test["ratio_test_predicted"] = computing_active_inactive_ratio(only_pred, prot)
        ratios_pred_test["strategy"] = strategy
        ratios_pred_test["fold"] = str(fold)
        ratios_pred_list.append(ratios_pred_test)

    return ratios_pred_list, pred_test


def computing_metrics_per_prot_bsl(prot, pred_test):
    """Computing metrics per protein for the random baseline"""
    dict_prot = {}
    predictions_test_sub = pred_test[pred_test["DeepAffinity Protein ID"] == prot]
    if predictions_test_sub.shape[0] == 0:
        return None
    dict_prot["acc"] = metrics.accuracy_score(y_true=predictions_test_sub["y_test"].values, 
                                 y_pred=predictions_test_sub["label"].values)
    try: 
        dict_prot["auroc"] = metrics.roc_auc_score(y_true = predictions_test_sub["y_test"].values, 
                                               y_score = predictions_test_sub["random_baseline"].values)
    except ValueError:
        print("Only one class present in y_true.")
        dict_prot["auroc"] = np.nan
    dict_prot["f1"] = metrics.f1_score(y_true=predictions_test_sub["y_test"].values, 
                                       y_pred=predictions_test_sub["label"].values)
    dict_prot["DeepAffinity Protein ID"] = prot
    dict_prot["balanced_acc"] = metrics.balanced_accuracy_score(y_true=predictions_test_sub["y_test"].values, 
                                                        y_pred=predictions_test_sub["label"].values)
    dict_prot["mcc"] = metrics.matthews_corrcoef(y_true=predictions_test_sub["y_test"].values, 
                                         y_pred=predictions_test_sub["label"].values)
    return dict_prot
  
def labelling(row):
    """Labelling samples in which random_baseline result is >= 0.5 as 1 (Actives) and the rest as 0 (inactives)"""
    if row["random_baseline"] >= 0.5:
        return 1
    else:
        return 0
