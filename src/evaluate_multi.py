import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_model_multi(model, loader, batch_size, crit, use_cuda, num_labels, bs):
    model.eval()
    data_size = len(loader)
    correct = 0
    total = 0
    correct = 0
    total = 0
    total_loss = 0
    true_pos = np.zeros(num_labels)
    true_neg = np.zeros(num_labels)
    false_pos = np.zeros(num_labels)
    false_neg = np.zeros(num_labels)
    last_pred = None
    last_y = None
    avg_loss = []
    for i, batch in enumerate(loader):
        if batch[0].size(0) != batch_size:
            continue
        model.eval()
        word_hidden = model.word_rnn.init_hidden(batch[0].size(0) * batch[0].size(1), True)
        sent_hidden = model.sent_rnn.init_hidden(True)
        if use_cuda:
            word_hidden, sent_hidden = word_hidden.cuda(), sent_hidden.cuda()

        batch_x, batch_y = Variable(batch[0], volatile=True), Variable(batch[1])
        length_x = batch[2].type(torch.FloatTensor).unsqueeze(1)
        length_x = Variable(length_x)
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            length_x = length_x.cuda()

        out = model(batch_x, word_hidden, sent_hidden, length_x)
        loss = crit(out, batch_y)
        total_loss += loss.data[0]
        avg_loss.append(loss.data[0])
        pred_prob = F.sigmoid(out)
        predicted = pred_prob.data > 0.5
        predicted = predicted.int()
        last_pred = predicted
        y = batch_y.data.int()
        last_y = y
        total += batch[0].size(0)
        correct += (predicted == y).sum()
        predicted = predicted.cpu().numpy()
        y = y.cpu().numpy()
        for k in range(y.shape[1]):
          true_pos[k] += np.sum((predicted[:,k][np.where(y[:,k] == 1)] == y[:,k][np.where(y[:,k] == 1)]))
          true_neg[k] += np.sum((predicted[:,k][np.where(y[:,k] == 0)] == y[:,k][np.where(y[:,k] == 0)]))
          false_pos[k] += np.sum((predicted[:,k][np.where(y[:,k] == 1)] != y[:,k][np.where(y[:,k] == 1)]))
          false_neg[k] += np.sum((predicted[:,k][np.where(y[:,k] == 0)] != y[:,k][np.where(y[:,k] == 0)]))
        if i == 1:
            print(pred_prob, predicted.data, y)
        if i % 1 == 0:
            print("Processed {} batches".format(i))
            break
    avg_loss = np.mean(avg_loss)
    micro_precision = np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_pos))
    micro_recall = np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_neg))
    micro_F = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
    print("correct: {}, avg_loss = {}, data_size: {}, prediction egs: {}, out egs: {}".format(
              correct / float(total), avg_loss, data_size * bs, last_pred[:5], last_y[:5]))
    print("True pos: {}, True neg: {}, False pos: {}, False neg: {}".format(
              np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg)))
    print("Micro precision: {:4f}, micro recall: {:4f}, micro F1: {:4f}".format(micro_precision, micro_recall, micro_F))
    return(avg_loss, (correct / float(total)),  micro_F, micro_precision, micro_recall)
