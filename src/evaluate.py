import torch
from torch.autograd import Variable

def evaluate(model, loader, batch_size, crit, use_cuda):
    model.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if batch[0].size(0) != batch_size:
            continue
        x = Variable(batch[0])
        y = Variable(batch[1])
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = crit(out, y)
        predicted = out.data > 0.5
        total += batch[1].size(0)
        correct += (predicted.float() == y.data).sum()
        if i % 20 == 0:
            print("Processed {} batches".format(i))
    print("% correct: {}, loss: {}, predicted: {}".format(
        correct / float(total), loss.data[0], predicted[:10]
    ))
