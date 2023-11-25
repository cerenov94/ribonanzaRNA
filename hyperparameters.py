import torch
class Config:
    seq_length = 457
    #kfold = KFold(n_splits= 9,shuffle = True,random_state = 42)
    batch_size = 64
    try:
        ckp = torch.load('logs/best_logs.pth')
    except:
        pass



config = Config()
