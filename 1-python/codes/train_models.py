import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import encode_onehot
import matplotlib.pyplot as plt
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer, AvgNonZeroReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
import config

def train_triplet_Vae_confidence(task, cnn, decoder, confidence, optimizer_s, h_dim, num_epochs, NumClasses, train_loader, test_loader):
    epochs = []

    train_acc = []
    train_loss = []
    train_vae_loss = []
    train_xentropy_loss = []
    train_confP2_loss = []
    train_confidence_loss = []
    train_triplet_loss = []
    train_error = []
    
    test_acc = []
    test_loss = []
    test_vae_loss = []
    test_xentropy_loss = []
    test_triplet_loss = []
    test_error = []

    lambda1 = 0.1
    lambda2 = 1
    lambda3 = 0.015
    reg = 0.001

    # prediction_criterion = nn.NLLLoss().cuda()
    prediction_criterion = nn.NLLLoss()

    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    distance = LpDistance(power=2)
    triplet_criterion = losses.TripletMarginLoss(distance=distance, reducer=AvgNonZeroReducer(), embedding_regularizer=LpRegularizer())
    # triplet_criterion = losses.TripletMarginLoss(distance=distance, reducer=AvgNonZeroReducer(), embedding_regularizer=LpRegularizer()).cuda()

    for epoch in range(num_epochs):
        cnn.train()
        decoder.train()
        confidence.train()
        epochs.append(epoch)
        # optimizer = lr_scheduler1(optimizer_s, lrate, epoch)
        optimizer = optimizer_s

        train_batch_ctr = 0.0
        running_loss = 0.0
        VAEloss_sum = 0.0
        xentropy_loss_sum = 0.0
        confidence_part2_loss_sum = 0.0
        confidence_loss_sum = 0.0
        triplet_loss_sum = 0.0

        correct_count = 0.0
        total = 0.0
        total_num = 0.0

        progress_bar = tqdm(train_loader)
        for i, (image, label) in enumerate(progress_bar):

            progress_bar.set_description('Epoch ' + str(epoch + 1))
            # image, label = Variable(image.cuda()), Variable(label.cuda())
            image, label = Variable(image), Variable(label)

            labels_onehot = Variable(encode_onehot(label, NumClasses))
            # optimizer.zero_grad()
            cnn.zero_grad()
            decoder.zero_grad()
            confidence.zero_grad()

            mu, logvar, laten_code, centers, distance, outputs = cnn(image)

            # vae loss
            # recon loss
            x_recon = decoder(laten_code)
            # reconloss = F.binary_cross_entropy(x_recon, image, reduction='sum')
            # reconloss = F.binary_cross_entropy(x_recon, image, reduction='mean')
            reconloss = torch.mean(torch.square(image - x_recon).sum(dim=1))
            # KL loss
            # KLD_element = mu.pow(2).add_(
            #     logvar.exp()).mul_(-1).add_(1).add_(logvar)
            # KL_loss = torch.sum(KLD_element).mul_(-0.5)
            KL_loss = torch.mean(0.5 * (torch.square(mu) + logvar.exp() - logvar - 1).sum(dim=1))
            VAEloss = reconloss + KL_loss
            VAEloss_sum += VAEloss.item()

            # triplet loss
            # hard_pairs = miner(laten_code, label)
            # triplet_loss = triplet_criterion(laten_code, label, hard_pairs)
            hard_pairs = miner(mu, label)
            triplet_loss = triplet_criterion(mu, label, hard_pairs)
            triplet_loss_sum += triplet_loss.item()

            # conf loss
            pred_original, confiden = confidence(laten_code)
            # pred_original, confiden = confidence(mu)
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
            xentropy_loss_sum += xentropy_loss.item()
            lmbda = 0.1
            confidence_loss = torch.mean(-torch.log(confiden))
            confidence_part2_loss_sum += confidence_loss.item()
            confest_loss = xentropy_loss + (lmbda * confidence_loss)
            confidence_loss_sum += confest_loss.item()
            if 0.3 > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif 0.3 <= confidence_loss.item():
                lmbda = lmbda / 0.99

            # total loss
            loss = lambda1 * VAEloss + lambda2 * confest_loss + lambda3 * triplet_loss
            total += loss.item()

            loss.backward()
            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1

            running_loss += loss.item()

            # for confidence
            pred_idx = torch.max(pred_original.data, 1)[1]
            total_num += label.size(0)
            correct_count += (pred_idx == label.data).sum()
            
            confaccuracy = correct_count / total_num
            progress_bar.set_postfix(
                VAEloss_avg='%.3f' % (VAEloss_sum / (i + 1)),
                xentropy_loss_avg='%.3f' % (xentropy_loss_sum / (i+1)),
                triplet_loss_avg='%.3f' % (triplet_loss_sum / (i+1)),
                total_loss_avg='%.3f' % (total / (i + 1)),
                confacc='%.3f' % confaccuracy)

        tqdm.write(
            'Train corrects: %.1f, Train samples: %.1f, Train accuracy: %.4f' %
            (correct_count, (total_num), confaccuracy))
        train_acc.append(confaccuracy)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(
            ((total_num) - correct_count) / (total_num))
        train_vae_loss.append(VAEloss_sum / train_batch_ctr)
        train_xentropy_loss.append(xentropy_loss_sum / train_batch_ctr)
        train_confP2_loss.append(confidence_part2_loss_sum / train_batch_ctr)
        train_confidence_loss.append(confidence_loss_sum / train_batch_ctr)
        train_triplet_loss.append(triplet_loss_sum / train_batch_ctr)

        cnn.eval()
        decoder.eval()
        confidence.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_VAEloss_sum = 0.0
        test_triplet_loss_sum = 0.0
        test_xentropy_loss_sum = 0.0

        correct_count = 0.0
        total = 0.0
        total_num = 0.0
        for image, label in test_loader:

            with torch.no_grad():
                image, label = Variable(image), Variable(label)
                # image, label = Variable(image.cuda()), Variable(label.cuda())
                mu, logvar, laten_code, centers, distance, outputs = cnn(image)
                # vae loss
                # recon loss
                x_recon = decoder(laten_code)
                # reconloss = F.binary_cross_entropy(x_recon, image, reduction='sum')
                # reconloss = F.binary_cross_entropy(x_recon, image, reduction='mean')
                reconloss = torch.mean(torch.square(image - x_recon).sum(dim=1))

                # KL loss
                # KLD_element = mu.pow(2).add_(
                #     logvar.exp()).mul_(-1).add_(1).add_(logvar)
                # KL_loss = torch.sum(KLD_element).mul_(-0.5)
                KL_loss = torch.mean(0.5 * (torch.square(mu) + logvar.exp() - logvar - 1).sum(dim=1))
                VAEloss = reconloss + KL_loss
                test_VAEloss_sum += VAEloss.item()

                # triplet loss
                # hard_pairs = miner(laten_code, label)
                # triplet_loss = triplet_criterion(laten_code, label, hard_pairs)
                hard_pairs = miner(mu, label)
                triplet_loss = triplet_criterion(mu, label, hard_pairs)
                test_triplet_loss_sum += triplet_loss.item()
                # conf loss
                # pred_original, confiden = confidence(laten_code)
                pred_original, confiden = confidence(mu)
                pred_original = F.softmax(pred_original, dim=-1)
                pred_new = torch.log(pred_original)
                xentropy_loss = prediction_criterion(pred_new, label)
                test_xentropy_loss_sum += xentropy_loss.item()
                # total loss
                loss = lambda1 * VAEloss + lambda2 * xentropy_loss + lambda3 * triplet_loss
                total += loss.item()
                test_running_loss += loss.item()

                test_batch_ctr = test_batch_ctr + 1

                # for confidence
                pred_idx = torch.max(pred_original.data, 1)[1]
                total_num += label.size(0)
                pred_idx = torch.max(pred_original.data, 1)[1]
                test_running_corrects += (pred_idx == label.data).sum()
                testconfaccuracy = test_running_corrects / total_num

        test_acc.append(testconfaccuracy)
        test_loss.append(1.0 * test_running_loss / test_batch_ctr)
        test_vae_loss.append(1.0 * test_VAEloss_sum / test_batch_ctr)
        test_xentropy_loss.append(test_xentropy_loss_sum / test_batch_ctr)
        test_triplet_loss.append(test_triplet_loss_sum / test_batch_ctr)

        tqdm.write(
            'Test corrects: %.1f, Test samples:%.1f, Test accuracy: %.4f' %
            (test_running_corrects, (total_num), testconfaccuracy))
        tqdm.write('Train loss: %.10f, Test loss: %.10f' %
                   (train_loss[epoch], test_loss[epoch]))
        tqdm.write('test xentropy loss: %.10f,  train xentropy loss: %.10f' %
                   (test_xentropy_loss[epoch], train_xentropy_loss[epoch]))
        tqdm.write('train confP2 loss: %.10f,  train confidence loss: %.10f' %
        (train_confP2_loss[epoch], train_confidence_loss[epoch]))
        tqdm.write('test Vae loss: %.10f,  train Vae loss: %.10f' %
                   (test_vae_loss[epoch], train_vae_loss[epoch]))
        tqdm.write('test triplet loss: %.10f,  train triplet loss: %.10f' %
                   (test_triplet_loss[epoch], train_triplet_loss[epoch]))
        tqdm.write('*' * 70)

    torch.save(cnn.state_dict(), '../model/{0}_final_model_{1}hidden_{2}epoches.pth'.format(task, h_dim, num_epochs))
    torch.save(confidence.state_dict(), '../model/{0}_final_conf_model_{1}hidden_{2}epoches.pth'.format(task, h_dim, num_epochs))
    torch.save(decoder.state_dict(), '../model/{0}_final_decoder_model_{1}hidden_{2}epoches.pth'.format(task, h_dim, num_epochs))

def update_confnet(confidence, optimizer_s, lrate, num_epochs, number_calsses, train_loader, dataset_train_len):
    epochs = []
    train_acc = []
    train_loss = []
    train_error = []
    train_xentropy_loss = []
    # prediction_criterion = nn.NLLLoss().cuda()
    prediction_criterion = nn.NLLLoss()

    for epoch in range(num_epochs):

        confidence.train()
        epochs.append(epoch)
        optimizer = optimizer_s
        train_batch_ctr = 0.0

        running_loss = 0.0
        confidence_loss_sum = 0.0
        xentropy_loss_sum = 0.0

        correct_count = 0.0
        total = 0.0
        total_num = 0.0

        progress_bar = tqdm(train_loader)
        for i, (image, label) in enumerate(progress_bar):

            progress_bar.set_description('Epoch ' + str(epoch + 1))
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
            xentropy_loss_sum += xentropy_loss.item()
            lmbda = 0.1
            confidence_loss = torch.mean(-torch.log(confiden))
            confest_loss = xentropy_loss + (lmbda * confidence_loss)
            confidence_loss_sum += confest_loss.item()
            if 0.3 > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif 0.3 <= confidence_loss.item():
                lmbda = lmbda / 0.99

            # total loss
            loss = confest_loss
            total += loss.item()

            loss.backward()
            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1

            running_loss += loss.item()

            # for confidence
            pred_idx = torch.max(pred_original.data, 1)[1]
            total_num += label.size(0)
            correct_count += (pred_idx == label.data).sum()
            epoch_acc = (float(correct_count) / (float(dataset_train_len)))
            confaccuracy = correct_count / total_num
            progress_bar.set_postfix(
                xentropy_loss_avg='%.3f' % (xentropy_loss_sum / (i+1)),
                confidence_loss_avg='%.3f' % (confidence_loss_sum / (i + 1)),
                total_loss_avg='%.3f' % (total / (i + 1)),
                confacc='%.3f' % confaccuracy)

        tqdm.write(
            'Train corrects: %.1f, Train samples: %.1f, Train accuracy: %.4f' %
            (correct_count, (dataset_train_len), epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(
            ((dataset_train_len) - correct_count) / (dataset_train_len))
        train_xentropy_loss.append(xentropy_loss_sum / train_batch_ctr)

        confidence.eval()
        tqdm.write('Train loss: %.10f' %
                   (train_loss[epoch]))
        tqdm.write('*' * 70)
    return confidence

def update_confnet_noprint(confidence, optimizer_s, lrate, num_epochs, number_calsses, train_loader, dataset_train_len):
    # epochs = []
    # train_acc = []
    # train_loss = []
    # train_error = []
    # train_xentropy_loss = []
    # # prediction_criterion = nn.NLLLoss().cuda()
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

def update_vae(cnn, decoder, confidence, optimizer_s, lrate,
                num_epochs, reg, number_calsses, train_loader, dataset_train_len,
                plotsFileName=None, csvFileName=None):
    epochs = []
    train_acc = []
    train_loss = []
    train_error = []
    train_gcpl_loss = []
    train_vae_loss = []
    train_xentropy_loss = []
    train_vae_loss = []
    prediction_criterion = nn.NLLLoss().cuda()
    for epoch in range(num_epochs):
        cnn.train()
        decoder.train()
        confidence.train()
        epochs.append(epoch)
        optimizer = optimizer_s
        running_loss = 0.0
        train_batch_ctr = 0.0

        VAEloss_sum = 0.0
        confidence_loss_sum = 0.0
        xentropy_loss_sum = 0.0
        correct_count = 0.0
        total = 0.0
        total_num = 0.0

        progress_bar = tqdm(train_loader)
        for i, (image, label) in enumerate(progress_bar):

            progress_bar.set_description('Epoch ' + str(epoch + 1))
            image, label = Variable(image.cuda()), Variable(label.cuda())
            labels_onehot = Variable(encode_onehot(label, number_calsses))
            # optimizer.zero_grad()
            cnn.zero_grad()
            decoder.zero_grad()
            confidence.zero_grad()

            mu, logvar, laten_code, centers, distance, outputs = cnn(image)
            _, preds = torch.max(distance, 1)
            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(laten_code, centers, label).squeeze()

            # recon loss
            x_recon = decoder(laten_code)
            recon_loss = F.binary_cross_entropy(x_recon,
                                                image,
                                                reduction='sum')
            
            # KL loss
            KLD_element = mu.pow(2).add_(
                logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KL_loss = torch.sum(KLD_element).mul_(-0.5)

            # conf loss
            pred_original, confiden = confidence(laten_code)
            pred_original = F.softmax(pred_original, dim=-1)
            confiden = torch.sigmoid(confiden)
            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confiden = torch.clamp(confiden, 0. + eps, 1. - eps)
            b = Variable(torch.bernoulli(torch.Tensor(confiden.size()).uniform_(0, 1))).cuda()
            conf = confiden * b + (1 - b)
            pred_new = pred_original * conf.expand_as(
                pred_original) + labels_onehot * (
                    1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)
            xentropy_loss = prediction_criterion(pred_new, label)
            xentropy_loss_sum += xentropy_loss.item()
            lmbda = 0.1
            confidence_loss = torch.mean(-torch.log(confiden))
            confest_loss = xentropy_loss + (lmbda * confidence_loss)
            confidence_loss_sum += confest_loss.item()
            if 0.8 > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif 0.8 <= confidence_loss.item():
                lmbda = lmbda / 0.99

            # total loss
            GCPLloss = 0.01 * (loss1 + reg * loss2)
            VAEloss = 0.01 * recon_loss + 0.01 * KL_loss
            VAEloss_sum += VAEloss.item()
            loss = GCPLloss + VAEloss + confest_loss
            total += loss.item()

            loss.backward()
            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1

            running_loss += loss.item()

            # for confidence
            pred_idx = torch.max(pred_original.data, 1)[1]
            total_num += label.size(0)
            correct_count += (pred_idx == label.data).sum()
            epoch_acc = (float(correct_count) / (float(dataset_train_len)))
            confaccuracy = correct_count / total_num
            progress_bar.set_postfix(
                vAEloss_avg='%.3f' % (VAEloss_sum / (i + 1)),
                xentropy_loss_avg='%.3f' % (xentropy_loss_sum / (i+1)),
                confidence_loss_avg='%.3f' % (confidence_loss_sum / (i + 1)),
                total_loss_avg='%.3f' % (total / (i + 1)),
                confacc='%.3f' % confaccuracy)

        tqdm.write(
            'Train corrects: %.1f, Train samples: %.1f, Train accuracy: %.4f' %
            (correct_count, (dataset_train_len), epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(
            ((dataset_train_len) - correct_count) / (dataset_train_len))
        train_vae_loss.append(VAEloss_sum / train_batch_ctr)
        train_xentropy_loss.append(xentropy_loss_sum / train_batch_ctr)

        cnn.eval()
        decoder.eval()
        confidence.eval()
        tqdm.write('Train loss: %.10f,  AE loss: %.10f' %
                   (train_loss[epoch], train_vae_loss[epoch]))
        tqdm.write('*' * 70)

def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance