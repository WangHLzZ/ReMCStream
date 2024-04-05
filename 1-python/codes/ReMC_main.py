from inspect import istraceback
from operator import mod, truediv
import numpy as np
import math
import os
import sys
import timeit
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from ReMC_models import EncoderVAE3, EncoderVAE2, EncoderVAE1,ConfidenceVae
from train_models import update_confnet, update_vae,update_confnet_noprint
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.multiprocessing as mp
import argparse
import config
import pandas as pd

from ast import Str
import torch.nn.functional as F
from CluStream import *
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
import ctypes
import copy
from tqdm import tqdm
from utils import CSVLogger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class LoadDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def GenerateSemiLable(Label, percent):
    LabelUsed = Label.copy()
    index = np.random.choice(np.arange(len(LabelUsed)),
                             size=int(percent * len(LabelUsed)),
                             replace=False)
    LabelUsed[index] = -1
    SemiLable = LabelUsed
    return SemiLable

def remc():
    setup_seed(args.seed)

    train_features = np.load('../datasets/{0}_features_{1}hidden_{2}epoch.npy'.format(args.datasetname, args_h, num_epochs))
    train_labels = np.load('../datasets/{0}_labels_{1}hiden_{2}epoch.npy'.format(args.datasetname, args_h, num_epochs))
    print(train_labels.shape)
    print(train_features.shape)

    df = pd.read_csv(config.dict_dataset[args.datasetname]['file'])
    X_all = np.array(df.iloc[:,:-1])
    y_all = np.array(df.iloc[:,-1])
    Clustream_features = X_all[config.NumInit:]
    Clustream_labels_original = y_all[config.NumInit:]
    Clustream_labels = GenerateSemiLable(Clustream_labels_original, 1-args.labelpercent)

    train_label_classes = [train_labels[np.where(
        train_labels == k)] for k in range(0, args_numclasses)]
    train_feature_classes = [train_features[np.where(
        train_labels == k)] for k in range(0, args_numclasses)]

    model_clu = CluStream(maxMcs=args.maxmcs, maxUMcs=args.maxumcs, nb_initial_points=args.eachclass, numK=args.numK)
    for i, train_feature_class in enumerate(train_feature_classes):
        if len(train_feature_class) > 0:
            model_clu.fit(train_feature_class, i)

    if feature_dim>= 256:
        model_extfea = EncoderVAE1(args_h, args.scale, args_numclasses, feature_dim)
    elif feature_dim <= 32 :
        model_extfea = EncoderVAE2(args_h, args.scale, args_numclasses, feature_dim)
    else :
        model_extfea = EncoderVAE3(args_h, args.scale, args_numclasses, feature_dim)
    model_extfea.load_state_dict(torch.load('../model/{0}_final_model_{1}hidden_{2}epoches.pth'.format(args.datasetname, args_h, num_epochs)))
    model_conf = ConfidenceVae(args_h, args_numclasses)
    model_conf.load_state_dict(torch.load('../model/{0}_final_conf_model_{1}hidden_{2}epoches.pth'.format(args.datasetname, args_h, num_epochs)))

    avg_radius = model_clu.calculate_avg_radius()
    progress_bar_mcs = tqdm(model_clu.labeled_micro_clusters)
    for i, microcluster in enumerate(progress_bar_mcs):
        progress_bar_mcs.set_description(
            'working on {} micro clusters '.format(len(model_clu.labeled_micro_clusters)))
        microcluster.confidence = 1
        microcluster.labelType = 1
        if microcluster.get_weight() == 1:
            microcluster.radius = avg_radius

    knn = KNeighborsClassifier(n_neighbors=args.numK)
    MCcenters = np.array(list((x.get_center())
                         for x in model_clu.labeled_micro_clusters), dtype=float)
    MClabels = np.array(list((x.get_labelMc())
                        for x in model_clu.labeled_micro_clusters), dtype=int)
    print(MCcenters.shape)
    print(MClabels.shape)
    knn.fit(MCcenters, MClabels)

    dataset_InitLatent = LoadDataset(train_features, train_labels)
    InitLatent_loader = torch.utils.data.DataLoader(
        dataset_InitLatent, batch_size=1, shuffle=False)
    Init_confidenceList = []
    for (latentFeature, latentlabel) in InitLatent_loader:
        with torch.no_grad():
            latentFeature, latentlabel = Variable(latentFeature), Variable(latentlabel)

            _ , Initconfidence = model_conf(latentFeature)
        Initconfidence_of_confnetwork = torch.sigmoid(Initconfidence)
        Initconfidence = Initconfidence_of_confnetwork.data.squeeze(0).cpu().numpy()
        Init_confidenceList = Init_confidenceList + list(Initconfidence)
    c_meanInit = sum(Init_confidenceList)/len(Init_confidenceList)
    c_stdInit = np.var(Init_confidenceList)
    c_threshold = c_meanInit + c_stdInit

    BufferSize = 10*args_numclasses 
    confidence_list = copy.deepcopy(Init_confidenceList) 
    if len(confidence_list)>BufferSize:
        confidence_list = confidence_list[-BufferSize:]
        assert(len(confidence_list) == BufferSize)
    EachClassBufferSize = int(BufferSize / args_numclasses)

    labeledData_list = []
    for i in range(0, args_numclasses):
        labeledData_list.append(train_feature_classes[i].tolist())
    for i in range(0, args_numclasses):
        if len(labeledData_list[i]) > EachClassBufferSize:
            labeledData_list[i] = labeledData_list[i][-EachClassBufferSize:]
            assert(len(labeledData_list[i]) == EachClassBufferSize)

    dataset_CluStream = LoadDataset(Clustream_features, Clustream_labels)
    CluStream_loader = torch.utils.data.DataLoader(
        dataset_CluStream, batch_size=1, shuffle=False)
    AccNowCount = 0
    DataNowCount = 0
    CluStreamAcc = 0.0

    lrate = 0.001
    tau = args.tau

    ConfnetNBbeforeUpdate = 0
    PlabelAccNowCount = 0
    PlabelNBCount = 0
    PlabelAccNowCountOnlyUnlabel = 0
    PlabelNBCountOnlyUnlabel = 0
    scorelist = []

    progress_bar_singledata = tqdm(CluStream_loader)
    for i, (image, label) in enumerate(progress_bar_singledata):
        DataNowCount += 1
        model_clu.timestamp += 1
        cpulabel = label

        with torch.no_grad():
            image, label = Variable(image), Variable(label)
            latencodeClu, logvarClu, z_sample, centersClu, distanceClu, outputsClu = model_extfea(image)

        # confidence
        predict_of_confnetwork, conf_of_confnetwork = model_conf(latencodeClu)
        predict_of_confnetwork = F.softmax(predict_of_confnetwork, dim=-1)
        featuresUsed = latencodeClu.data.cpu().numpy().flatten()
        pred_idx = torch.max(predict_of_confnetwork.data, 1)[1]
        predict_idx = pred_idx.data.cpu().numpy()[0]
        confiden_of_confnetwork = torch.sigmoid(conf_of_confnetwork)
        conf_of_confnetwork_used = confiden_of_confnetwork.data.cpu().numpy()[0]

        micro_clusters = model_clu.labeled_micro_clusters.copy()
        micro_clusters_2 = model_clu.unlabeled_micro_clusters.copy()
        micro_clusters += micro_clusters_2
        closest_cluster, dis = model_clu.find_closest_cluster(
            featuresUsed, micro_clusters)

        flag = model_clu.check_fit_in_cluster(featuresUsed, closest_cluster)

        score = 0.0
        UpdataConfKMCsL = False
        UpdataConfKMCsL1 = False
        if flag and (closest_cluster.get_labelMc() != -1):
            predictKnn = closest_cluster.get_labelMc()
            if cpulabel != -1:
                UpdataConfKMCsL1 = True
            if (predictKnn == predict_idx):
                if conf_of_confnetwork_used >= c_threshold:
                    eT = model_clu.timestamp - closest_cluster.update_timestamp
                    part1 = math.exp(-0.001* eT)
                    part2 = 1
                    part3 = 1 /(1 + math.exp(-1 * closest_cluster.confidence))
                    factor = 1
                    score = factor * part1 * part2 * part3
        else:
            predictKnn = knn.predict(latencodeClu.data.cpu().numpy())
            IndexOfpredictMCs = knn.kneighbors(latencodeClu.data.cpu().numpy(), n_neighbors=args.numK, return_distance=True)
            predictKnn = predictKnn[0]
            knnConMcs = model_clu.labeled_micro_clusters
            if cpulabel != -1:
                UpdataConfKMCsL = True
            if (predictKnn == predict_idx):
                if conf_of_confnetwork_used >= c_threshold:
                    for k in range(args.numK):
                        eT = model_clu.timestamp - knnConMcs[IndexOfpredictMCs[1][0][k]].update_timestamp
                        part1 = math.exp(-0.01* eT)
                        if model_clu.check_fit_in_cluster(featuresUsed, knnConMcs[IndexOfpredictMCs[1][0][k]]):
                            part2 = 1
                        else:                            
                            a = knnConMcs[IndexOfpredictMCs[1][0][k]]
                            if (a.get_radius()==0) | (distance.euclidean(featuresUsed , a.get_center())==0):
                                assert(a.get_radius()==0)
                                print(knnConMcs[IndexOfpredictMCs[1][0][k]].get_radius(), distance.euclidean(featuresUsed, knnConMcs[IndexOfpredictMCs[1][0][k]].get_center()))
                                part2 = (knnConMcs[IndexOfpredictMCs[1][0][k]].get_radius())/(distance.euclidean(featuresUsed, knnConMcs[IndexOfpredictMCs[1][0][k]].get_center()))
                            part2 = (knnConMcs[IndexOfpredictMCs[1][0][k]].get_radius())/(distance.euclidean(featuresUsed, knnConMcs[IndexOfpredictMCs[1][0][k]].get_center()))
                        part3 = 1 /(1 + math.exp(-1 * knnConMcs[IndexOfpredictMCs[1][0][k]].confidence))
                        factor = 1 if (knnConMcs[IndexOfpredictMCs[1][0][k]].get_labelMc() == predictKnn) else -1
                        score += factor * part1 * part2 * part3
                    score = score / args.numK

        scorelist.append(score)
        if score >= tau:
            p_label = predictKnn
            PlabelNBCount += 1
            if cpulabel == -1:
                PlabelNBCountOnlyUnlabel += 1
            if p_label == Clustream_labels_original[i]:
                PlabelAccNowCount += 1
                if cpulabel == -1:
                    PlabelAccNowCountOnlyUnlabel += 1

        else:
            p_label = -1
        
        if cpulabel != -1:
            
            if len(labeledData_list[cpulabel]) >= EachClassBufferSize:
                labeledData_list[cpulabel].pop(0)
            labeledData_list[cpulabel].append(list(featuresUsed))
            for classindex in range(0, args_numclasses):
                assert(len(labeledData_list[classindex]) <= EachClassBufferSize)

            if (predictKnn == predict_idx):
                if (cpulabel==predictKnn) and (score<tau):
                    tau = tau * (1 - 0.01)
                elif (cpulabel!=predictKnn) and (score>=tau):
                    tau = tau * (1 + 0.01)

        labelforMCs = cpulabel
        correctKmcs = 0

        if UpdataConfKMCsL1:
            if closest_cluster.get_labelMc() != labelforMCs:
                closest_cluster.confidence -= 1
            else:
                closest_cluster.confidence += 1
        if UpdataConfKMCsL:
            for k in range(args.numK):
                if knnConMcs[IndexOfpredictMCs[1][0][k]].get_labelMc() != labelforMCs:
                    knnConMcs[IndexOfpredictMCs[1][0][k]].confidence -= 1/args.numK 
                else:
                    knnConMcs[IndexOfpredictMCs[1][0][k]].confidence += 1
                    correctKmcs += 1
        
        for x in model_clu.labeled_micro_clusters:
            x.confidence = x.confidence * math.pow(2, - (1e-4) * 0.1 *(model_clu.timestamp - x.update_timestamp))
        for x in model_clu.unlabeled_micro_clusters:
            x.confidence = x.confidence * math.pow(2, - (1e-4) * 0.1 * (model_clu.timestamp - x.update_timestamp))
        
        mcs_tobedel = [] 
        if len(model_clu.labeled_micro_clusters)>args.numK:
            for cluster in model_clu.labeled_micro_clusters:
                if cluster.get_confidence() <= 0.000001:
                    mcs_tobedel.append(cluster)
            if len(mcs_tobedel) <= (len(model_clu.labeled_micro_clusters)-args.numK):
                for cluster in mcs_tobedel:
                    model_clu.labeled_micro_clusters.remove(cluster)

            mcs_tobedel = []
        for cluster in model_clu.unlabeled_micro_clusters:
            if cluster.get_confidence() <= 0.000001:
                model_clu.unlabeled_micro_clusters.remove(cluster)

        
        micro_clusters = model_clu.labeled_micro_clusters.copy()
        micro_clusters_2 = model_clu.unlabeled_micro_clusters.copy()
        micro_clusters += micro_clusters_2
        closest_cluster, dis = model_clu.find_closest_cluster(
            featuresUsed, micro_clusters)
        
        flag = model_clu.check_fit_in_cluster(featuresUsed, closest_cluster)
        
        if flag:
            if cpulabel != -1:
                if closest_cluster.get_labelMc() != -1:
                    if closest_cluster.labelType == 0:
                        if (cpulabel != closest_cluster.get_labelMc()):
                            model_clu.labeled_micro_clusters.remove(closest_cluster)
                            model_clu.creat(featuresUsed, (label.data.cpu().numpy())[0])
                        else:
                            closest_cluster.insert(featuresUsed, model_clu.timestamp)
                            closest_cluster.labelType = 1
                    else:
                        if (cpulabel != closest_cluster.get_labelMc()):
                            if dis < (closest_cluster.get_radius()/3):
                                model_clu.labeled_micro_clusters.remove(closest_cluster)
                            model_clu.creat(featuresUsed, (label.data.cpu().numpy())[0])
                        else:
                            closest_cluster.insert(featuresUsed, model_clu.timestamp)
                else:
                    closest_cluster.insert(featuresUsed, model_clu.timestamp)
                    closest_cluster.labelMc = (label.data.cpu().numpy())[0]
                    closest_cluster.labelType = 1
                    model_clu.unlabeled_micro_clusters.remove(closest_cluster)
                    model_clu.labeled_micro_clusters.append(closest_cluster)
            elif p_label != -1:
                if closest_cluster.get_labelMc() != -1:
                    if closest_cluster.labelType == 0:
                        if (cpulabel != closest_cluster.get_labelMc()):
                            model_clu.labeled_micro_clusters.remove(closest_cluster)
                            model_clu.creat(featuresUsed, (label.data.cpu().numpy())[0], p_label)
                        else:
                            closest_cluster.insert(featuresUsed, model_clu.timestamp)
                    else:
                        if (cpulabel != closest_cluster.get_labelMc()):                            
                            model_clu.creat(featuresUsed, (label.data.cpu().numpy())[0], p_label)
                        else:
                            closest_cluster.insert(featuresUsed, model_clu.timestamp)
                else:                    
                    closest_cluster.insert(featuresUsed, model_clu.timestamp)
                    closest_cluster.labelMc = p_label
                    closest_cluster.labelType = 0

                    model_clu.unlabeled_micro_clusters.remove(closest_cluster)
                    model_clu.labeled_micro_clusters.append(closest_cluster)
            else: 
                closest_cluster.insert(featuresUsed, model_clu.timestamp)
        else:
            if (label.data.cpu().numpy()[0] == -1) and (p_label != -1):
                    model_clu.creat(featuresUsed, label.data.cpu().numpy()[0], p_label)
            else:
                model_clu.creat(featuresUsed, (label.data.cpu().numpy())[0])
        
        if (predictKnn == Clustream_labels_original[i]):
            AccNowCount += 1
        FlagUpdateConfnet = False
        if cpulabel != -1:
            if (predict_idx != Clustream_labels_original[i]):
                # nb_confnet_error += 1
                if conf_of_confnetwork_used >= c_threshold:
                    FlagUpdateConfnet = True

        CluStreamAcc = AccNowCount/DataNowCount

        del knn
        knn = KNeighborsClassifier(n_neighbors=args.numK)
        knnMcs = model_clu.labeled_micro_clusters
        labelnum = len(model_clu.labeled_micro_clusters)
        unlabelnum = len(model_clu.unlabeled_micro_clusters)
        totalnum = labelnum + unlabelnum

        MCcenters = np.array(list((x.get_center())
                             for x in knnMcs), dtype=float)
        MClabels = np.array(list((x.get_labelMc())
                            for x in knnMcs), dtype=int)
        knn.fit(MCcenters, MClabels)

        ConfnetNBbeforeUpdate += 1
        if FlagUpdateConfnet == True:
            labeledData = []
            labeledDataLabel = []
            for classindex in range(0, args_numclasses):
                labeledData = labeledData + labeledData_list[classindex]
                labeledDataLabel = labeledDataLabel + [classindex]*len(labeledData_list[classindex])
            train_data = np.array(labeledData)
            train_label = np.array(labeledDataLabel)

            update_train = LoadDataset(train_data, train_label)
            update_loader = torch.utils.data.DataLoader(update_train, batch_size=32, shuffle=True)
            update_train_len = 1.0 * len(update_train)
            parameters_update = list(model_conf.parameters())
            optimizer_update = torch.optim.Adam(parameters_update, lr = lrate)

            # model_conf = update_confnet(model_conf, optimizer_update, 0.001, 5, args_numclasses, update_loader, update_train_len)
            model_conf = update_confnet_noprint(model_conf, optimizer_update, 0.001, 5, args_numclasses, update_loader, update_train_len)

            ConfnetNBbeforeUpdate = 0
            FlagUpdateConfnet = False


        if len(confidence_list)>=BufferSize:
            confidence_list.pop(0)
            confidence_list = confidence_list + list(conf_of_confnetwork_used)
            assert(len(confidence_list) == BufferSize)
        else:
            confidence_list = confidence_list + list(conf_of_confnetwork_used)
        c_mean = sum(confidence_list)/len(confidence_list)
        c_std = np.var(confidence_list)
        c_threshold = c_mean + c_std

        progress_bar_singledata.set_postfix(acc='%.16f' % CluStreamAcc)
        
    
if __name__ == '__main__':

    os.chdir(sys.path[0])
    mp.set_start_method('spawn')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', '-dn', default="spam")
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--numK', type=int, default=1)
    parser.add_argument('--labelpercent', type=float, default=0.15)
    parser.add_argument('--tau', type=float, default=0.4)
    # parser.add_argument('--cthreshold', type=float, default=0.9)
    parser.add_argument('--maxmcs', type=int, default=1000)
    parser.add_argument('--maxumcs', type=int, default=1000)
    parser.add_argument('--eachclass', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--scale',
                        type=float,
                        default=2,
                        help='scaling factor for distance')
    parser.add_argument('--reg',
                        type=float,
                        default=0.001,
                        help='regularization coefficient')
    args, _ = parser.parse_known_args()
    num_epochs = config.dict_dataset[args.datasetname]['n_epoch']
    args_h = config.dict_dataset[args.datasetname]['h_dim']
    args_numclasses = config.dict_dataset[args.datasetname]['NumClasses']
    feature_dim = config.dict_dataset[args.datasetname]['featuredim']
    remc()