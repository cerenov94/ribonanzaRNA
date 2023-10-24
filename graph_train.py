import numpy as np
import torch
from model import GNN
from graph_dataset import Graph_Dataset,Test_Bpps_Dataset,Valid_Dataset,Test_Graph_Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.nn as gnn
from hyperparameters import config
from time import time
import datetime as dtime
import pandas as pd
import gc
from torch_geometric.data import Batch
import torch_geometric as tg
import torch_geometric.transforms as T
from tqdm import tqdm
from model_2 import Gat_NN
from model_3 import AttentiveFP


x_train = pd.read_parquet('train_files/clean_train.parquet')
x_valid = pd.read_parquet('train_files/valid_with_structure.parquet')

#transform = T.Compose([T.LocalDegreeProfile()])
train_ds = Graph_Dataset(x_train)
valid_ds = Valid_Dataset(x_valid)




train_loader = DataLoader(train_ds,batch_size=config.batch_size,pin_memory=True,drop_last=True,shuffle=True,num_workers=8)
valid_loader = DataLoader(valid_ds,batch_size=config.batch_size,pin_memory=True,drop_last=False,shuffle=False,num_workers=8)
del x_train,x_valid
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = GNN()
#model = AttentiveFP(4,256,2,edge_dim=7,num_layers=8,num_timesteps=12,dropout=0.1)




def loss_fn(pred, target):
    loss = F.l1_loss(pred, target.clip(0,1), reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss

def weighted_rmse_loss(pred, target, weight):

    loss = F.mse_loss(pred[~torch.isnan(target)],target[~torch.isnan(target)].clip(0,1),reduction='none')
    loss = (1/weight) * loss.mean()
    return torch.sqrt(loss)



criterion = torch.nn.L1Loss(reduction='none')

def learn(model,train_loader,valid_loader,loss_fn,resume = False):

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0005)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=5)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    last_epoch = 0
    best_train_loss =float('inf')
    best_valid_loss = float('inf')
    best_compet_score = float('inf')
    model.to(device)
    if resume:
        print('Loading last checkpoint...')
        model.load_state_dict(config.ckp['weights'])
        optimizer.load_state_dict(config.ckp['optimizer'])
        best_train_loss = config.ckp['best_train_loss']
        best_valid_loss = config.ckp['best_valid_loss']
        best_compet_score = config.ckp['best_compet_score']
        last_epoch = config.ckp['epoch']
    print('train step...\t')
    print(f'Starting train from: Epoch: {last_epoch} | Best valid loss : {best_valid_loss:.5f} | compet score: {best_compet_score:.5f}\t')

    for epoch in range(last_epoch,1000):
        start_time = time()

        model.train()
        train_loss = 0


        for batch in tqdm(train_loader): 

            optimizer.zero_grad()
            batch = batch.to(device)
            prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
            unbatched_prediction = tg.utils.unbatch(prediction, batch.batch, dim=0)

            new_preds = torch.concatenate([torch.nn.functional.pad(x, pad=[0, 0, 0, 206 - x.size(0)]) for x in unbatched_prediction])
            pred_a,pred_d = new_preds[:,0].unsqueeze(dim=-1),new_preds[:,1].unsqueeze(dim=-1)
            target_a,target_d = batch['y'][:,0].unsqueeze(dim=-1),batch['y'][:,1].unsqueeze(dim=-1)


            loss = (loss_fn(pred_a,target_a) + loss_fn(pred_d,target_d))/2
            # loss = (weighted_rmse_loss(pred_a,target_a,batch['noise_a'].unsqueeze(dim=-1)) +
            #         weighted_rmse_loss(pred_d,target_d,batch['noise_d'].unsqueeze(dim=-1)))/2
            loss.sum().backward()
            optimizer.step()
            train_loss += loss.sum().item()



        train_loss = train_loss/len(train_loader)
        train_duration = str(dtime.timedelta(seconds=time() - start_time))[:7]


        model.eval()
        valid_loss = 0
        compet_score = 0
        print('valid step...\t')
        with ((torch.inference_mode())):
            for batch in tqdm(valid_loader):
                batch = batch.to(device)
                prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch)

                unbatched_prediction = tg.utils.unbatch(prediction, batch.batch, dim=0)

                new_preds = torch.concatenate(
                    [torch.nn.functional.pad(x, pad=[0, 0, 0, 206 - x.size(0)]) for x in unbatched_prediction])
                pred_a, pred_d = new_preds[:, 0].unsqueeze(dim=-1), new_preds[:, 1].unsqueeze(dim=-1)
                target_a, target_d = batch['y'][:, 0].unsqueeze(dim=-1), batch['y'][:, 1].unsqueeze(dim=-1)
                loss = (loss_fn(pred_a, target_a) + loss_fn(pred_d, target_d)) / 2
                # loss = (weighted_rmse_loss(pred_a, target_a, batch['noise_a'].unsqueeze(dim=-1)) +
                #         weighted_rmse_loss(pred_d, target_d, batch['noise_d'].unsqueeze(dim=-1))) / 2

                valid_loss += loss.sum().item()

                score = (loss_fn(pred_a, target_a) + loss_fn(pred_d, target_d)) / 2
                compet_score += score.item()
            compet_score /= len(valid_loader)
            valid_loss = valid_loss/len(valid_loader)

        scheduler.step(compet_score)
        print(f"Epoch {epoch + 1}: Train duration {train_duration} | Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f} | compet score: {compet_score:.5f}")
        # saving model
        if valid_loss < best_valid_loss or compet_score < best_compet_score:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            if compet_score < best_compet_score:
                best_compet_score = compet_score

            torch.save({'weights': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch+1,
                            'best_train_loss': best_train_loss,
                            'best_valid_loss': best_valid_loss,
                            'best_compet_score': best_compet_score,
                            },f'logs/train_logs.pth')
            print('Train logs saved.')
            torch.cuda.empty_cache()
            gc.collect()
        if (epoch+1) % 16 == 0:
            print('Creating submission...\t')

            test_seq = pd.read_parquet('train_files/test_seq_struct.parquet')
            test_ds = Test_Graph_Dataset(test_seq)
            test_dataloader = tg.loader.DataLoader(test_ds, batch_size=16, drop_last=False, num_workers=8)

            model.eval()
            preds = []
            with torch.inference_mode():
                for batch in tqdm(test_dataloader):
                    batch = batch.to(device)
                    prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch)

                    preds.append(prediction.detach().cpu().numpy())

            preds = np.concatenate(preds, dtype=np.float32)

            submission = pd.DataFrame({'reactivity_DMS_MaP': preds[:, 1], 'reactivity_2A3_MaP': preds[:, 0]})
            submission = submission.clip(0, 1, axis=0)
            submission = submission.reset_index().rename(columns={'index': 'id'})
            submission.to_parquet(f'logs/submission_{epoch+1}_valid_metric:{compet_score:.5f}_valid_loss:{valid_loss:.5f}.parquet')
            del test_seq, test_ds,test_dataloader,submission,preds
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    learn(model,train_loader,valid_loader,loss_fn,resume=True)
