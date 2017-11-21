
def evaluate(model, loader, batch_size, crit, use_cuda):
    import torch
    from torch.autograd import Variable
    correct = 0
    total = 0

    for batch in loader:
        if batch[0].size(0) != batch_size:
            continue
        x = Variable(batch[0])
        y = Variable(batch[1].view(-1))
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        #hidden = model.init_hidden()
        x = model(x)#, hidden)
        loss = crit(x, y)
        _, predicted = torch.max(x.data, 1)
        total += batch[1].size(0)
        correct += (predicted == y.data).sum()
      
    return (correct / float(total), loss.data[0], predicted[:10])


'''
def evaluate(model, loader, batch_size, label_map):
    correct = 0
    total = 0

    for i in loader:
        if len(i[0]) != batch_size:
            continue
        seq_len = len(i[0][0])
        batch_x = np.ndarray((batch_size, seq_len, 400))
        for bs, adm in enumerate(i[0]):
            for idx, note in enumerate(adm):
                note_vec = []
                for token in note:
                    note_vec.append(pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']))
                
                batch_x[bs, idx, :] = np.mean(note_vec)  
                
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(np.array([label_map[_] for _ in i[1]])).long().view(batch_size)
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
        x = Variable(batch_x)
        hidden = model.init_hidden()
        x = model(x, hidden)
        _, predicted = torch.max(x.data, 1)
        total += batch_y.size(0)
        print predicted
        print batch_y
        correct += (predicted == batch_y.data).sum()
        break
    return correct / float(total)

'''

