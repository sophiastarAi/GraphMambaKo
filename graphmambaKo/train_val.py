from torch import nn

from tools import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.optim import lr_scheduler
import util
import time
mpl.use('Agg')

###################################### train #######################################
def train_batch(dataloader, model, scaler, optimizer, device, criterion, steps, steps_back, backward, lamb, nu, eta,
                gradclip,adj,edge_index,edge_attr,mode_cons):
    train_loss = 0
    count=0
    print_every = 20
    loss = util.masked_mae

    train_loss = []
    train_mape = []
    train_mae = []
    train_time = []
    t1 = time.time()

    dataloader['train_loader'].shuffle()
    for batch_idx, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
    # for batch_idx, data_list in enumerate(train_loader):
        count+=1
        model.train()
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)

        out, out_back = model(trainx,edge_index,edge_attr, mode='forward')
        out = scaler.inverse_transform(out[0])


        metrics = util.metric(out, trainy)
        loss_fwd = torch.sqrt(criterion(out, trainy))
        train_loss.append(metrics[2])
        train_mape.append(metrics[1])
        train_mae.append(metrics[0])
        if batch_idx % print_every == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train MAE: {:.4f}'
            print(log.format(batch_idx, train_loss[-1], train_mape[-1], train_mae[-1]), flush=True)

        loss_bwd = 0.0
        loss_consist = 0.0
        if backward == 1:

            out, out_back = model(data_list[-1].to(device),edge_index,edge_attr, mode='backward')

            for k in range(steps_back):
                if k == 0:
                    loss_bwd = criterion(out_back[k], data_list[::-1][k + 1].to(device))
                else:
                    loss_bwd += criterion(out_back[k], data_list[::-1][k + 1].to(device))

            if mode_cons:
                I = torch.eye(K,K).to(device)
                A_I= model(I, mode='Linear')
                eigenvalue=torch.linalg.eigvals(A)
                freq = get_freq_torch(eigenvalue)
                freq =torch.sort(freq,descending =True).values
                freq_real = torch.from_numpy(np.array([1 / 2.5, 1 / 5.5,1 / 7.5, 1 / 12, 1 / 24]))
                loss_mode = cal_mode_loss(freq,freq_real).to(device)

                mo = torch.from_numpy(np.ones(K))
                loss_mode2= torch.sum(mo-torch.abs(eigenvalue))

                loss = loss_fwd + lamb * loss_identity + 100 * loss_mode+50*loss_mode2
            else:
                loss = loss_fwd + nu * loss_bwd + lamb * loss_identity

        loss = loss_fwd

            # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
        optimizer.step()

    t2 = time.time()
    train_time.append(t2 - t1)
    return loss

def val_batch(dataloader, model, scaler, device, criterion, steps, adj, edge_index,edge_attr):
    val_loss = 0
    model.eval()

    for batch_idx, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        # model.train()
        model.eval()
        valx = torch.Tensor(x).to(device)
        valy = torch.Tensor(y).to(device)

        out, out_back = model(valx,edge_index,edge_attr, mode='forward')
        out = scaler.inverse_transform(out[0])

        metrics = util.metric(out, valy)
        loss_fwd = metrics[2]
        val_loss += loss_fwd
    val_loss /= (batch_idx+1)
    # print("val loss", val_loss)
    return val_loss

def test_batch(dataloader, model, device, criterion, steps, adj, edge_index,edge_attr):
    val_loss = np.zeros(steps)
    total_loss = 0
    model.eval()

    for batch_idx, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    # for batch_idx, data_list in enumerate(test_loader):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        model.eval()

        out, out_back = model(testx,edge_index,edge_attr, mode='forward')
        loss_fwd = nn.MSELoss().to(device)(out[0], testy)
        val_loss[0] += nn.MSELoss().to(device)(out[0], testy).item()
        total_loss+=loss_fwd.item()
    val_loss /= batch_idx + 1
    total_loss /= batch_idx + 1
    return np.sqrt(val_loss),total_loss

def train(model, modelpath, dataloader, logger, scaler, lr,gamma, weight_decay,
          lamb, num_epochs, learning_rate_change, epoch_update,early_stop,adj,edge_index,edge_attr,
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, gradclip=1, mode_cons=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print(optimizer)
    device = get_device()
    criterion = nn.MSELoss().to(device)
    best_epoch = 0
    val_loss_min = 1000000
    train_loss_epoch=[]
    val_loss_epoch = []

    for epoch in range(num_epochs):
        train_loss = train_batch(dataloader, model, scaler, optimizer, device, criterion, steps, steps_back, backward, lamb, nu, eta,
                gradclip,adj,edge_index,edge_attr,mode_cons)
        val_loss = val_batch(dataloader, model, scaler, device, criterion, steps,adj,edge_index,edge_attr)
        # rmse,test_loss = test_batch(dataloader, model, device, criterion, steps, adj, edge_index, edge_attr)
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        scheduler.step()

        if val_loss < val_loss_min:
            # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
            val_loss_min = val_loss
            best_epoch = epoch
            print(f'best epoch: {best_epoch}, val loss: {val_loss}, Minimum val loss!!!!')
            torch.save(model.state_dict(), modelpath + '/model'+'.pkl')

        if (epoch) % 1 == 0:
            batch_x = range(0, len(train_loss_epoch))
            print('-------------------- Epoch %s ------------------' % (epoch + 1))
            print(f"average train loss: {train_loss}")
            print(f"average val loss: {val_loss}")
            # print(f"rmse on val: {rmse}")
            if hasattr(model.dynamics, 'dynamics'):
                w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
    return model
