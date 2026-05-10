import torch
import numpy as np
import config
from ReMC_models import EncoderVAE3, EncoderVAE2, EncoderVAE1,ConfidenceVae
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from train_models import update_confnet
import copy

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

def encoderandconf_predict(data2predict, datasetname, numUpdate):
    # setup_seed(2024)
    num_epochs = config.dict_dataset[datasetname]['n_epoch']
    args_h = config.dict_dataset[datasetname]['h_dim']
    args_numclasses = config.dict_dataset[datasetname]['NumClasses']
    feature_dim = config.dict_dataset[datasetname]['featuredim']
    scale = 2
    model_path = '../model/{0}_final_model_{1}hidden_{2}epoches.pth'.format(
        datasetname, args_h, num_epochs)
    if feature_dim>= 256:
        model = EncoderVAE1(args_h, scale, args_numclasses, feature_dim)
    elif feature_dim <= 32 :
        model = EncoderVAE2(args_h, scale, args_numclasses, feature_dim)
    else :
        model = EncoderVAE3(args_h, scale, args_numclasses, feature_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(data2predict).reshape(1,-1)
        # print(input_tensor)
        output = model(input_tensor)

    if numUpdate==0:
        confnet_path = '../model/{0}_final_conf_model_{1}hidden_{2}epoches.pth'.format(
            datasetname, args_h, num_epochs)
    else:
        confnet_path = '../model_update/{0}_updated_conf_model.pth'.format(datasetname)
    confnet = ConfidenceVae(args_h, args_numclasses)
    confnet.load_state_dict(torch.load(confnet_path))
    confnet.eval()
    with torch.no_grad():
        hidden_tensor = torch.tensor(output[0])
        # print(hidden_tensor)
        predict_of_confnetwork, conf_of_confnetwork = confnet(hidden_tensor)
    predict_of_confnetwork = F.softmax(predict_of_confnetwork, dim=-1)

    pred_idx = torch.max(predict_of_confnetwork.data, 1)[1]
    predict_idx = pred_idx.data.cpu().numpy()[0]
    confiden_of_confnetwork = torch.sigmoid(conf_of_confnetwork)
    conf_of_confnetwork_used = confiden_of_confnetwork.data.cpu().numpy()[0]
    return output[0].squeeze(0).numpy(),predict_idx,conf_of_confnetwork_used

def confnet_predict(lowfeature2predict,labels, datasetname):
    num_epochs = config.dict_dataset[datasetname]['n_epoch']
    args_h = config.dict_dataset[datasetname]['h_dim']
    args_numclasses = config.dict_dataset[datasetname]['NumClasses']
    confnet_path = '../model/{0}_final_conf_model_{1}hidden_{2}epoches_{3}_{4}0.pth'.format(
        datasetname, args_h, num_epochs)
    confnet = ConfidenceVae(args_h, args_numclasses)
    confnet.load_state_dict(torch.load(confnet_path))
    confnet.eval()

    dataset_InitLatent = LoadDataset(lowfeature2predict, labels)
    InitLatent_loader = torch.utils.data.DataLoader(
        dataset_InitLatent, batch_size=1, shuffle=False)
    Init_confidenceList = []
    for (latentFeature, latentlabel) in InitLatent_loader:
        with torch.no_grad():
            latentFeature, latentlabel = Variable(latentFeature.cuda()), Variable(latentlabel.cuda())
            _ , Initconfidence = confnet(latentFeature)
        Initconfidence_of_confnetwork = torch.sigmoid(Initconfidence)
        Initconfidence = Initconfidence_of_confnetwork.data.squeeze(0).cpu().numpy()
        # Init_confidenceList.append(Initconfidence)
        Init_confidenceList = Init_confidenceList + list(Initconfidence)
    # c_threshold = sum(Init_confidenceList)/len(Init_confidenceList)
    c_meanInit = sum(Init_confidenceList)/len(Init_confidenceList)
    c_stdInit = np.var(Init_confidenceList)
    c_threshold = c_meanInit + c_stdInit


    return c_threshold, Init_confidenceList

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    out,out2,out3 = encoderandconf_predict(torch.from_numpy(np.random.rand(1,8)).float(),'elec')
    print('a')
    print(out,out2,out3)
