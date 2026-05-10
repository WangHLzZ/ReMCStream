import torch
import numpy as np
import config
from ReMC_models import ConfidenceVae

def confnet_predict(lowfeature2predict,labels, datasetname):

    num_epochs = config.dict_dataset[datasetname]['n_epoch']
    args_h = config.dict_dataset[datasetname]['h_dim']
    args_numclasses = config.dict_dataset[datasetname]['NumClasses']
    confnet_path = '../model/{0}_final_conf_model_{1}hidden_{2}epoches.pth'.format(datasetname, args_h, num_epochs)
    confnet = ConfidenceVae(args_h, args_numclasses)
    confnet.load_state_dict(torch.load(confnet_path))
    confnet.eval()
    Init_confidenceList = []
    for i in range(lowfeature2predict.shape[0]):
        with torch.no_grad():
            hidden_tensor = torch.tensor(lowfeature2predict[i,:])
            _ , Initconfidence = confnet(hidden_tensor)
        Initconfidence_of_confnetwork = torch.sigmoid(Initconfidence)
        Initconfidence = Initconfidence_of_confnetwork.data.squeeze(0).cpu().numpy()
        # print(Initconfidence)
        Init_confidenceList.append(Initconfidence.item())
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
    print('a')
    
