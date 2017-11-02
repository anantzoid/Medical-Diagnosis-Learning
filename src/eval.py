
def evaluate(model, loader, batch_size):
    model.eval()
    correct = 0
    total = 0
    for i in loader:

    for batch in loader:
        if batch[0].size(0) != batch_size:
            continue
        x = Variable(batch[0])
        hidden = model.init_hidden()
        x = model(x, hidden)
        _, predicted = torch.max(x.data, 1)
        total += batch[1].size(0)
        correct += (predicted == batch[1]).sum()
      
    return correct / float(total)
