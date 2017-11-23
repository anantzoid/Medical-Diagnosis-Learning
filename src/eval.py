def evaluate(model, loader, batch_size, crit, use_cuda):
    import torch
    from torch.autograd import Variable
    correct = 0
    total = 0

    for batch in loader:
        if batch[0].size(0) != batch_size:
            continue
        x = Variable(batch[0])
        y = Variable(batch[1])#.view(-1))
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        #hidden = model.init_hidden()
        x = model(x)#, hidden)
        loss = crit(x, y)
        predicted = x.data > 0.5
        total += batch[1].size(0)
        correct += (predicted.float() == y.data).sum()
    return (correct / float(total), loss.data[0], predicted[:10])
