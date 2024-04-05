import numpy as np
import os
import sys
import timeit
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from ReMC_models import ConfidenceVae, DecoderVAE3, DecoderVAE2, DecoderVAE1, EncoderVAE3, EncoderVAE2, EncoderVAE1
from train_models import train_triplet_Vae_confidence
import pandas as pd
# import torch.multiprocessing as mp
import argparse
import config
import torch.nn.functional as F

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
def train_extract(input_datasetname):
    print(input_datasetname)
    setup_seed(2022)
    print(os.getcwd())
    # os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', '-dn', default="spam")
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scale',
                        type=float,
                        default=2,
                        help='scaling factor for distance')
    parser.add_argument('--reg',
                        type=float,
                        default=.001,
                        help='regularization coefficient')
    args, _ = parser.parse_known_args()
    args.datasetname = input_datasetname
    num_epochs = config.dict_dataset[args.datasetname]['n_epoch']
    args_h = config.dict_dataset[args.datasetname]['h_dim']
    args_numclasses = config.dict_dataset[args.datasetname]['NumClasses']
    feature_dim = config.dict_dataset[args.datasetname]['featuredim']

    df = pd.read_csv(config.dict_dataset[args.datasetname]['file'])
    X_all = np.array(df.iloc[:,:-1])
    y_all = np.array(df.iloc[:,-1])

    train_x = X_all[:config.NumInit]
    test_x = X_all[config.NumInit:config.NumInit+1000]
    train_y = y_all[:config.NumInit]
    test_y = y_all[config.NumInit:config.NumInit+1000]

    # sclar = MinMaxScaler(feature_range=(0,1), copy=True)

    train_num = train_x.shape[0]
    test_num = test_x.shape[0]
    print(f'train_number: {train_num}')
    print(train_x.shape)
    print(f'test_number: {test_num}')
    print(test_x.shape)
    assert (feature_dim==train_x.shape[1])
    assert (args_numclasses==len(set(list(y_all))))
    dataset_train = LoadDataset(train_x, train_y)
    dataset_test = LoadDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    dataset_test_len = 1.0 * len(dataset_test)
    dataset_train_len = 1.0 * len(dataset_train)
    choose = 0

    if feature_dim>= 256:
        model = EncoderVAE1(args_h, args.scale, args_numclasses, feature_dim)
        decoder = DecoderVAE1(args_h, feature_dim, args_numclasses)
    elif feature_dim <= 32 :
        model = EncoderVAE2(args_h, args.scale, args_numclasses, feature_dim)
        decoder = DecoderVAE2(args_h, feature_dim, args_numclasses)
    else :
        model = EncoderVAE3(args_h, args.scale, args_numclasses, feature_dim)
        decoder = DecoderVAE3(args_h, feature_dim, args_numclasses)

    confidence = ConfidenceVae(args_h, args_numclasses)


    parameters = list(model.parameters()) + list(decoder.parameters())+ list(confidence.parameters())
    optimizer_s = torch.optim.Adam(parameters, lr = args.lr)
    print(choose)
    # 训练
    train_triplet_Vae_confidence(args.datasetname, model, decoder, confidence, optimizer_s, args_h,
                    num_epochs, args_numclasses, train_loader, test_loader)
    # 提取特征
    nb_netupdate = 0
    dataset_train = LoadDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=1,
                                            shuffle=False)
    dataset_train_len = 1.0 * len(dataset_train)


    if feature_dim>= 256:
        model = EncoderVAE1(args_h, args.scale, args_numclasses, feature_dim)
    elif feature_dim <= 32 :
        model = EncoderVAE2(args_h, args.scale, args_numclasses, feature_dim)
    else :
        model = EncoderVAE3(args_h, args.scale, args_numclasses, feature_dim)

    model.load_state_dict(torch.load('../model/{0}_final_model_{1}hidden_{2}epoches.pth'.format(args.datasetname, args_h, num_epochs)))

    extracted_features = []
    true_labels = []
    i = 0
    for (image, label) in train_loader:
        i += 1
        my_embedding = torch.zeros(1, args_h)

        with torch.no_grad():
            image, label = Variable(image), Variable(label)
            latentcode = model(image)
        my_embedding = latentcode[0].squeeze(0)
        my_embedding = my_embedding.detach().cpu().numpy()
        extracted_features.append(my_embedding)
        true_labels.append(label.cpu().data.numpy()[0])
    np.save('../datasets/{0}_features_{1}hidden_{2}epoch'.format(args.datasetname, args_h, num_epochs), extracted_features)
    np.save('../datasets/{0}_labels_{1}hiden_{2}epoch'.format(args.datasetname, args_h, num_epochs), true_labels)
    # del model
    print(feature_dim)