import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_model(model, loader, batch_size, crit, use_cuda):
    model.eval()
    correct = 0
    total = 0
    f1, precision, recall = [], [], []
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
        avg_loss.append(loss.data[0])
        pred_prob = F.softmax(out)
        predicted = _, predicted = torch.max(pred_prob.data, 1)
        total += batch[1].size(0)
        # print(predicted, batch_y)
        correct += (predicted == batch_y.data).sum()
        f1.append(f1_score( val_batch_y.data.cpu().numpy(), predicted.cpu().numpy(), average='micro'))
        precision.append(precision_score(val_batch_y.data.cpu().numpy(),predicted.cpu().numpy(),  average='micro'))
        recall.append(recall_score(predicted.cpu().numpy(), val_batch_y.data.cpu().numpy(), average='micro'))

        if i == 1:
            print(pred_prob, predicted.cpu().numpy(), batch_y.data.cpu().numpy())
        if i % 100 == 0:
            print("Processed {} batches".format(i))
    avg_loss = np.mean(avg_loss)
    f1 = np.mean(f1)
    precision = np.mean(precision)
    recall = np.mean(recall)
    print("{}/{} correct, {:.3f}%, avg loss: {}, avg f1: {}, avg precision: {}, avg recall: {}".format(
        correct, total, (correct / float(total)), avg_loss, f1, precision, recall))
    return(avg_loss, (correct / float(total)),  f1, precision, recall)
