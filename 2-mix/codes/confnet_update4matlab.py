import torch
import numpy as np
import config
from ReMC_models import ConfidenceVae
# from train_models import update_confnet_noprint
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import encode_onehot

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
    
def confnet_update(labelData4train,labels, datasetname, numUpdate):

    # setup_seed(2024)
    lrate = 0.001
    num_epochs = config.dict_dataset[datasetname]['n_epoch']
    args_h = config.dict_dataset[datasetname]['h_dim']
    args_numclasses = config.dict_dataset[datasetname]['NumClasses']

    if numUpdate==0:
        confnet_path = '../model/{0}_final_conf_model_{1}hidden_{2}epoches.pth'.format(
            datasetname, args_h, num_epochs)
    else:
        confnet_path = '../model_update/{0}_updated_conf_model.pth'.format(datasetname)
    confnet = ConfidenceVae(args_h, args_numclasses)
    confnet.load_state_dict(torch.load(confnet_path))
    save_path = '../model_update/{0}_updated_conf_model.pth'.format(datasetname)

    train_data = np.array(labelData4train)
    train_label = np.array(labels)
    update_train = LoadDataset(train_data, train_label)
    update_loader = torch.utils.data.DataLoader(update_train, batch_size=32, shuffle=True)
    update_train_len = 1.0 * len(update_train)
    parameters_update = list(confnet.parameters())
    optimizer_update = torch.optim.Adam(parameters_update, lr = lrate)

    model_conf = update_confnet_noprint(confnet, optimizer_update, 0.001, 5, args_numclasses, update_loader, update_train_len)
    torch.save(model_conf.state_dict(), save_path)

    # return lrate

def update_confnet_noprint(confidence, optimizer_s, lrate, num_epochs, number_calsses, train_loader, dataset_train_len):
    prediction_criterion = nn.NLLLoss()

    for epoch in range(num_epochs):

        confidence.train()
        # epochs.append(epoch)
        optimizer = optimizer_s
        # train_batch_ctr = 0.0

        # running_loss = 0.0
        # confidence_loss_sum = 0.0
        # xentropy_loss_sum = 0.0

        # correct_count = 0.0
        # total = 0.0
        # total_num = 0.0

        # progress_bar = tqdm(train_loader)
        # for i, (image, label) in enumerate(progress_bar):
        for i, (image, label) in enumerate(train_loader):

            # progress_bar.set_description('Epoch ' + str(epoch + 1))
            image, label = Variable(image), Variable(label)
            # image, label = Variable(image.cuda()), Variable(label.cuda())

            labels_onehot = Variable(encode_onehot(label, number_calsses))
            # optimizer.zero_grad()
            confidence.zero_grad()

            # conf loss
            pred_original, confiden = confidence(image)
            pred_original = F.softmax(pred_original, dim=-1)
            confiden = torch.sigmoid(confiden)
            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confiden = torch.clamp(confiden, 0. + eps, 1. - eps)
            # b = Variable(
            #     torch.bernoulli(torch.Tensor(confiden.size()).uniform_(
            #         0, 1))).cuda()
            b = Variable(
                torch.bernoulli(torch.Tensor(confiden.size()).uniform_(
                    0, 1)))
            conf = confiden * b + (1 - b)
            pred_new = pred_original * conf.expand_as(
                pred_original) + labels_onehot * (
                    1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)
            xentropy_loss = prediction_criterion(pred_new, label)
            # xentropy_loss_sum += xentropy_loss.item()
            lmbda = 0.1
            confidence_loss = torch.mean(-torch.log(confiden))
            confest_loss = xentropy_loss + (lmbda * confidence_loss)
            # confidence_loss_sum += confest_loss.item()
            if 0.3 > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif 0.3 <= confidence_loss.item():
                lmbda = lmbda / 0.99

            # total loss
            loss = confest_loss
            # total += loss.item()

            loss.backward()
            optimizer.step()
        #     train_batch_ctr = train_batch_ctr + 1

        #     running_loss += loss.item()

        #     # for confidence
        #     pred_idx = torch.max(pred_original.data, 1)[1]
        #     total_num += label.size(0)
        #     correct_count += (pred_idx == label.data).sum()
        #     epoch_acc = (float(correct_count) / (float(dataset_train_len)))
        #     confaccuracy = correct_count / total_num
        #     progress_bar.set_postfix(
        #         xentropy_loss_avg='%.3f' % (xentropy_loss_sum / (i+1)),
        #         confidence_loss_avg='%.3f' % (confidence_loss_sum / (i + 1)),
        #         total_loss_avg='%.3f' % (total / (i + 1)),
        #         confacc='%.3f' % confaccuracy)

        # tqdm.write(
        #     'Train corrects: %.1f, Train samples: %.1f, Train accuracy: %.4f' %
        #     (correct_count, (dataset_train_len), epoch_acc))
        # train_acc.append(epoch_acc)
        # train_loss.append(1.0 * running_loss / train_batch_ctr)
        # train_error.append(
        #     ((dataset_train_len) - correct_count) / (dataset_train_len))
        # train_xentropy_loss.append(xentropy_loss_sum / train_batch_ctr)

        confidence.eval()
        
        # tqdm.write('Train loss: %.10f' %
        #            (train_loss[epoch]))
        # tqdm.write('*' * 70)
    return confidence