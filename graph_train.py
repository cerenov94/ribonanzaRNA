import numpy as np
import torch
from graph_dataset import Train_Dataset_with_additional_data,Valid_Dataset_177,Test_Graph_Dataset,Graph_Dataset,Valid_Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.nn as gnn
from hyperparameters import config
import pandas as pd
import gc
from torch_geometric.data import Batch
import torch_geometric as tg
import torch_geometric.transforms as T
from tqdm import tqdm
from model_2 import AttentiveGraphNet

import warnings
warnings.filterwarnings('ignore')

import torch_geometric.transforms as T

x_train = pd.read_parquet('train_files/train_with_structure.parquet')
x_valid = pd.read_parquet('train_files/valid_with_structure.parquet')



#transform = T.Compose([T.AddSelfLoops(),T.AddRandomWalkPE(walk_length=4,attr_name=None),T.AddLaplacianEigenvectorPE(k=4,attr_name='x')])
transform = T.AddRandomWalkPE(walk_length=8,attr_name='pe')
train_ds = Graph_Dataset(x_train,transform=transform)
#train_ds  = Valid_Dataset(x_valid)
valid_ds = Valid_Dataset(x_valid,transform=transform)




train_loader = DataLoader(train_ds,batch_size=config.batch_size,pin_memory=True,drop_last=True,shuffle=True,num_workers=8)
valid_loader = DataLoader(valid_ds,batch_size=config.batch_size,pin_memory=True,drop_last=False,shuffle=False,num_workers=8)
del x_train,x_valid
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#model = GNN(in_channels=12,hidden_channels=128,decoder_hidden=128,edge_dim=6,dropout=0.5,num_layers=12,num_attentive_layers=6)
#model = AttentiveFP(6,512,2,edge_dim=6,num_layers=8,num_timesteps=2,dropout=0.5)
model = AttentiveGraphNet(in_channels=4,pe=8,hidden_channels=192,out_channels=2,edge_dim=5,num_layers=8,num_a_layers=8,dropout=0.1)
#model = GraphTransformer(12,192,2,5,12,4)


def loss_fn(pred, target):
    loss = F.l1_loss(pred, target.clip(0,1), reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss




criterion = torch.nn.L1Loss(reduction='none')

def learn(model,train_loader,valid_loader,loss_fn,resume = False):

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=2,factor=0.5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    last_epoch = 0
    best_train_loss =float('inf')
    best_valid_loss = float('inf')
    model.to(device)
    if resume == 'best':
        print('Loading last checkpoint...')
        model.load_state_dict(config.ckp['weights'])
        optimizer.load_state_dict(config.ckp['optimizer'])
        #optimizer.param_groups[0]['lr'] = 0.0005
        best_train_loss = config.ckp['best_train_loss']
        best_valid_loss = config.ckp['best_valid_loss']

        last_epoch = config.ckp['epoch']

    elif resume == 'last':
        ckp = torch.load('logs/current_logs.pth')
        model.load_state_dict(ckp['weights'])
        optimizer.load_state_dict(ckp['optimizer'])
        best_train_loss = ckp['best_train_loss']
        best_valid_loss = config.ckp['best_valid_loss']

        last_epoch = ckp['epoch']

    print('Current LR',optimizer.param_groups[0]['lr'])
    print(f'Starting train from: Epoch: {last_epoch} | Best valid loss : {best_valid_loss:.5f} \t')
    for epoch in range(last_epoch,1000):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,batch.pe)
            unbatched_prediction = tg.utils.unbatch(prediction, batch.batch, dim=0)
            new_preds = torch.concatenate([torch.nn.functional.pad(x, pad=[0, 0, 0, 206 - x.size(0)]) for x in unbatched_prediction])

            pred_a, pred_d = new_preds[:, 0].unsqueeze(dim=-1), new_preds[:, 1].unsqueeze(dim=-1)
            target_a, target_d = batch['y'][:, 0].unsqueeze(dim=-1), batch['y'][:, 1].unsqueeze(dim=-1)

            loss = (loss_fn(pred_a,target_a) + loss_fn(pred_d,target_d))/2
            loss.sum().backward()
            #print(loss)


            optimizer.step()
            train_loss += loss.sum().item()
        train_loss = train_loss/len(train_loader)


        model.eval()
        valid_loss = 0
        with ((torch.inference_mode())):
            for batch in tqdm(valid_loader):
                batch = batch.to(device)
                prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,batch.pe)
                unbatched_prediction = tg.utils.unbatch(prediction, batch.batch, dim=0)
                new_preds = torch.concatenate(
                    [torch.nn.functional.pad(x, pad=[0, 0, 0, 206 - x.size(0)]) for x in unbatched_prediction])
                pred_a, pred_d = new_preds[:, 0].unsqueeze(dim=-1), new_preds[:, 1].unsqueeze(dim=-1)
                target_a, target_d = batch['y'][:, 0].unsqueeze(dim=-1), batch['y'][:, 1].unsqueeze(dim=-1)

                loss = (loss_fn(pred_a, target_a) + loss_fn(pred_d, target_d)) / 2
                valid_loss += loss.sum().item()
            valid_loss = valid_loss/len(valid_loader)


        scheduler.step(valid_loss)
        print(f"Epoch {epoch + 1}| Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f} ")
        # saving model
        torch.save({'weights': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_train_loss': train_loss,
                    'best_valid_loss':valid_loss,
                    }, f'logs/current_logs.pth')
        if valid_loss < best_valid_loss:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            torch.save({'weights': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch+1,
                            'best_train_loss': best_train_loss,
                            'best_valid_loss': best_valid_loss,
                            },f'logs/best_logs.pth')

            print('Train logs saved.')
            torch.cuda.empty_cache()
            gc.collect()
        if (epoch+1) % 100 == 0:
            print('Creating submission...\t')

            test_seq = pd.read_parquet('train_files/test_seq_struct.parquet')
            test_ds = Test_Graph_Dataset(test_seq)
            test_dataloader = tg.loader.DataLoader(test_ds, batch_size=config.batch_size//2, drop_last=False, num_workers=8)

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
            #submission.to_parquet(f'logs/submission_{epoch+1}_valid_loss:{valid_loss:.5f}.parquet')
            submission.to_csv(f'logs/submission_{epoch + 1}_valid_loss:{valid_loss:.5f}.csv',index = False)
            del test_seq, test_ds,test_dataloader,submission,preds
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    learn(model,train_loader,valid_loader,loss_fn)
