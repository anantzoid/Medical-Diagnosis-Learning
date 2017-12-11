import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def eval_model(model, loader, batch_size, crit, use_cuda):
    model.eval()
    correct = 0
    total = 0
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
        if i == 1:
            print(pred_prob, predicted, batch_y)
        if i % 100 == 0:
            print("Processed {} batches".format(i))
    print("{}/{} correct, {:.3f}%, avg loss: {}".format(
        correct, total, (correct / float(total)), np.mean(avg_loss)))
