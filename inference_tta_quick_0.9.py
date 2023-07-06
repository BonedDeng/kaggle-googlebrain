#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
NUM_WORKERS = 8

RESULT_PATH='/home/bowen/kaggle/googlebrain/result/'
DATA_PATH = "/home/bowen//kaggle/googlebrain/data/"  

sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_test = pd.read_csv(DATA_PATH + 'test.csv')


# In[3]:

def preprocess(df):

    r_map = {5: 0.5, 20: 2, 50: 5}
    c_map = {10: 1, 20: 2, 50: 5}
    df['R'] = df['R'].map(r_map)
    df['C'] = df['C'].map(c_map)

    df['u_in_cumsum'] = df['u_in'].groupby(df['breath_id']).cumsum() / 200
    
    df['logu_in'] = np.log1p(df['u_in'])
    for col in ['time_step','u_in','logu_in']:
        df[f'{col}_diff'] = df[f'{col}']-df.groupby("breath_id")[f'{col}'].shift(1)
        df[f'{col}_diff2'] = df[f'{col}_diff']-df.groupby("breath_id")[f'{col}_diff'].shift(1)
        df[f'{col}_diff3'] = df[f'{col}_diff2']-df.groupby("breath_id")[f'{col}_diff2'].shift(1)
        df[f'{col}_diff4'] = df[f'{col}_diff3']-df.groupby("breath_id")[f'{col}_diff3'].shift(1)
        
#         df[f'{col}_diff'] = df[f'{col}']-df.groupby("breath_id")[f'{col}'].shift(-1)
#         df[f'{col}_diff2'] = df[f'{col}_diff']-df.groupby("breath_id")[f'{col}_diff'].shift(-1)
#         df[f'{col}_diff3'] = df[f'{col}_diff2']-df.groupby("breath_id")[f'{col}_diff2'].shift(-1)
#         df[f'{col}_diff4'] = df[f'{col}_diff3']-df.groupby("breath_id")[f'{col}_diff3'].shift(-1)
        
        
        
    
    def time_until_exhale(u_out_):
        zero_locs = np.where(u_out_ != 0)[0]
        if len(zero_locs) == 0:
            return np.arange(len(u_out_))[::-1]
        else:
            return np.clip(zero_locs[0] - np.arange(len(u_out_)), 0, None)
        
    u_out = df["u_out"].values.copy().reshape(-1, 80)
    for i in range(len(u_out)):
        u_out[i] = time_until_exhale(u_out[i])
        
    df["time_until_exhale"] = u_out.flatten()
    
    
    return df.fillna(-1)


# In[4]:


df_train = preprocess(df_train)
df_test = preprocess(df_test)

# give index for discrete pressure
unique_pressure = df_train["pressure"].unique()
unique_pressure.sort()

pressure_to_idx = {v: k for k, v in enumerate(unique_pressure)}
df_train["label"] = df_train["pressure"].map(pressure_to_idx)
n_label = len(pressure_to_idx)

df_train.loc[df_train['u_out']==1,"label"]=-100





# In[2]:



import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=4,dropout=0.1)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.GELU(),
            nn.Linear(logit_dim, num_classes),
        )
        
       #======分类任务============ 
        self.mlp2 = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.GELU(),
            nn.Linear(logit_dim, logit_dim),
            nn.GELU(),
            nn.Linear(logit_dim,950),
        )
        
        

        
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        features = self.lstm(x)[0]
        pred     = self.logits(features)
        cls_pred = self.mlp2(features)
        
        return pred,cls_pred
    
    
import torch
from torch.utils.data import Dataset

#=======================get the smilar_inex===================
def process_train_df(df):
    uin = {}
    pressure = {}
    for k, _ in df.groupby(['R','C']):
        uin[k] = _['u_in'].values.reshape(-1, 80)
        pressure[k] = _['pressure'].values.reshape(-1, 80)
    return uin, pressure

train_uin, train_pressure = process_train_df(df_train)

uins = df_test['u_in'].values.reshape(-1,80)
rcs = df_test[['R','C']].values[::80]
similar_index = []
for i in tqdm(range(len(rcs))):
    uin = uins[i]
    rc = rcs[i]
    cur_uin = train_uin[tuple(rc)]
    idx = np.argsort(np.abs(cur_uin-uin).mean(-1))[:5]
    similar_index.append((tuple(rc),idx))

#=======================Done===================

class VentilatorDataset(Dataset):
    def __init__(self, df, train_df, mode = 'train',select_idx=0):
        self.df = df
        self.select_idx=select_idx
        self.train_uin, self.train_pressure = train_uin, train_pressure
        
        if "pressure" not in self.df.columns:
            self.df['pressure'] = 0
            self.df['label']=0
        
        self.len = self.df.shape[0]//80
        self.mode = mode
        self.prepare_data()
        self.similar_index = similar_index
        
                
    def __len__(self):
        return self.len
    
    def prepare_data(self):
        self.pressures = np.split(self.df['pressure'].values, self.len)
        self.u_outs = np.split(self.df['u_out'].values, self.len)
        cols =[i for i in self.df.columns if i not in ['id','breath_id','pressure','label']]
        self.inputs =  np.split(self.df[cols].values, self.len)
        self.cls_label=np.split(self.df['label'].values, self.len)


    #================modify===================
    def __getitem__(self, idx):
        rc, index = self.similar_index[idx]
        #index = random.choice(index)
        index = index[self.select_idx]
        train_uin = self.train_uin[rc][index]
        train_uin_diff1 = np.append(-1, np.diff(train_uin))
        train_uin_diff2 = np.append([-1,-1], np.diff(np.diff(train_uin)))
        
        train_pressure = self.train_pressure[rc][index]
        train_pressure_diff1 = np.append(-1, np.diff(train_pressure))
        train_pressure_diff2 = np.append([-1,-1], np.diff(np.diff(train_pressure)))
        
        input = self.inputs[idx]
        train_uin_diff_diff = train_uin - input[:,3]
        train_uin_diff1_diff1 = train_uin_diff1 - input[:,11]
        train_uin_diff2_diff2 = train_uin_diff2 - input[:,12]
        
        input = np.concatenate([input,train_uin.reshape(-1,1),train_uin_diff1.reshape(-1,1), train_uin_diff2.reshape(-1,1),                                train_uin_diff_diff.reshape(-1,1), train_uin_diff1_diff1.reshape(-1,1), train_uin_diff2_diff2.reshape(-1,1),                                 train_pressure.reshape(-1,1),train_pressure_diff1.reshape(-1,1),train_pressure_diff2.reshape(-1,1)], axis = -1)
        
        diff1 = np.append(np.diff(self.pressures[idx]), [0])
        diff2 = np.append(np.diff(diff1), [0])
        diff_p = self.pressures[idx] - train_pressure
        
        
        data = {
            "input": torch.tensor(input, dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
            "pdiff": torch.tensor(diff1, dtype=torch.float),
            "pdiff2": torch.tensor(diff2, dtype=torch.float),
             "pdiff_p": torch.tensor(diff_p, dtype=torch.float),
            "class_label":torch.tensor(self.cls_label[idx], dtype=torch.long),
        }
        
        return data


# In[3]:


import os
import torch
import random
import numpy as np


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


# ### Metric & Loss
# > The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

# In[9]:


def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """
    
    y = np.array(np.split(df['pressure'].values, len(df)//80))
    w = 1 - np.array(np.split(df['u_out'].values, len(df)//80))
    
    if len(preds.shape) == 1:
        preds = preds.reshape(-1, 80)
    assert y.shape == preds.shape and w.shape == y.shape, (y.shape, preds.shape, w.shape)
    
    mae = w * np.abs(y - preds)
    mae = mae.sum() / w.sum()
    
    return mae



class VentilatorLoss(nn.Module):
    def __call__(self, preds, y, u_out, y_diff1, y_diff2,y_diffp):
        w = 1 - u_out
        mae = w * (y - preds[:, :, 0]).abs()
        mae = mae.sum(-1) / w.sum(-1)

        diff_mae = w * (y_diff1 - preds[:, :, 1]).abs() +  w * (y_diff2 - preds[:, :, 2]).abs()+ w * (y_diffp - preds[:, :, 3]).abs() 
        diff_mae = diff_mae.sum(-1) / w.sum(-1)
        

        return (mae + diff_mae) / 4
    
    
class CLSLoss(nn.Module):
    def __call__(self, cls_pred, class_label):
        cls_loss_fn=torch.nn.CrossEntropyLoss(ignore_index = -100)
        cls_loss=cls_loss_fn(cls_pred.reshape(-1, 950), class_label.flatten())

        return cls_loss


# In[4]:


import gc
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
#from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


def fit(
    model,
    train_dataset,
    val_dataset,
    loss_name="L1Loss",
    optimizer="Adam",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda",
    fold=0
):
    avg_val_loss = 0.
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    loss_fct = VentilatorLoss()
    #loss_fct = torch.nn.L1Loss()
    loss_cls=CLSLoss()



    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=Config.T0, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    best_mae = np.inf
    
    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0
        for step, data in enumerate(train_loader):
            with autocast():
                pred,cls_pred = model(data['input'].to(device))
                loss = 0.9*loss_fct(
                    pred,
                    data['p'].to(device),
                    data['u_out'].to(device),
                    data['pdiff'].to(device),
                    data['pdiff2'].to(device),
                    data['pdiff_p'].to(device),
                ).mean()+0.1*loss_cls(cls_pred,data['class_label'].to(device))
                
   
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avg_loss += loss.item() / len(train_loader)

            scheduler.step(epoch+step/len(train_loader))
            optimizer.zero_grad() 

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.4f}",
                end="\t",
            )
            
        if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
            model.eval()
            mae, avg_val_loss = 0, 0
            preds = []

            with torch.no_grad():
                for data in val_loader:
                    pred,cls_pred = model(data['input'].to(device))

                    loss = 0.9*loss_fct(
                        pred,
                        data['p'].to(device),
                        data['u_out'].to(device),
                        data['pdiff'].to(device),
                        data['pdiff2'].to(device),
                        data['pdiff_p'].to(device),
                    ).mean()+0.1*loss_cls(cls_pred,data['class_label'].to(device))
  
                    avg_val_loss += loss.item() / len(val_loader)
                    preds.append(pred[:,:,0].detach().cpu().numpy())
            preds = np.concatenate(preds, 0)
            mae = compute_metric(val_dataset.df, preds)

            print(f"val_loss={avg_val_loss:.4f}\tmae={mae:.4f}")
                
            if mae < best_mae:
                best_mae=mae
                save_model_weights(model,f'{Config.selected_model}_{fold}.pt',verbose=False,cp_folder=RESULT_PATH)

        else:
            print("")
                    

    del (val_loader, train_loader, loss, data, pred)
    gc.collect()
    torch.cuda.empty_cache()

    return preds


# ### Predict

# In[11]:


def predict(
    model,
    dataset,
    batch_size=64,
    device="cuda"
):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (PathologyDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    
    preds = []
    with torch.no_grad():
        for data in loader:
            pred,_ = model(data['input'].to(device))#.reshape(-1,2)
            preds.append(pred[:,:,0].detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    return preds


# ## Train

# In[12]:


def train(config, df_train, df_val, df_test, fold):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        df_test (pandas dataframe): Test metadata.
        fold (int): Selected fold.

    Returns:
        np array: Study validation predictions.
    """

    seed_everything(config.seed)

    model = RNNModel(
        input_dim=config.input_dim,
        lstm_dim=config.lstm_dim,
        dense_dim=config.dense_dim,
        logit_dim=config.logit_dim,
        num_classes=config.num_classes,
    ).to(config.device)
    model.zero_grad()

    train_dataset = VentilatorDataset(df_train, df_train, 'train')
    val_dataset = VentilatorDataset(df_val, df_train, 'valid')
    test_dataset = VentilatorDataset(df_test, df_train, 'test')

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training breathes")
    print(f"    -> {len(val_dataset)} validation breathes")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        loss_name=config.loss,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
        fold=fold
    )
    
    model.load_state_dict(torch.load(RESULT_PATH+f'{Config.selected_model}_{fold}.pt'))
    
    pred_test= predict(
        model, 
        test_dataset, 
        batch_size=config.val_bs, 
        device=config.device
    )

    del (model, train_dataset, val_dataset, test_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val, pred_test


# ### $k$-fold

# In[13]:


from sklearn.model_selection import GroupKFold

def k_fold(config, df, df_test):
    """
    Performs a patient grouped k-fold cross validation.
    """

    pred_oof = np.zeros(len(df))
    preds_test = []
    
    gkf = GroupKFold(n_splits=config.k)
    splits = list(gkf.split(X=df, y=df, groups=df["breath_id"]))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            pred_val, pred_test = train(config, df_train, df_val, df_test, i)
            
            pred_oof[val_idx] = pred_val.flatten()
            preds_test.append(pred_test.flatten())
            
    metric_value=compute_metric(df, pred_oof)
    print(f'\n -> CV MAE : {metric_value :.3f}')

    return metric_value,pred_oof, np.median(preds_test, 0)


# ## Main

# In[14]:


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    
    k = 20
    selected_folds =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
    
    #k = 10
    #selected_folds =[0,1,2,3,4,5,6,7,8,9] 
    
    # Model
    selected_model = 'lstm_mul_v4_10fold'
    input_dim = 29

    dense_dim = 512
    lstm_dim = 512
    logit_dim = 512
    num_classes = 4

    loss = "L1Loss"  # not used
    optimizer = "AdamW"
    batch_size = 128
    val_bs = 2048   # 1024
    epochs = 280  #280
    T0=40

    lr = 3e-4
    warmup_prop = 0
 
    first_epoch_eval = 100


# In[ ]:


# metric_value,pred_oof, pred_test = k_fold(
    # Config, 
    # df_train,
    # df_test,
# )





model = RNNModel(
        input_dim=Config.input_dim,
        lstm_dim=Config.lstm_dim,
        dense_dim=Config.dense_dim,
        logit_dim=Config.logit_dim,
        num_classes=Config.num_classes,
    ).to(Config.device)

preds = []
for fold in range(Config.k):
    print(f'=======================fold {fold}=======================')
    model.load_state_dict(torch.load(RESULT_PATH+f'{Config.selected_model}_{fold}.pt'))
    for i in range(5):
        test_dataset = VentilatorDataset(df_test,df_train,mode='test',select_idx=i)
        pred_test= predict(
            model, 
            test_dataset, 
            batch_size=Config.val_bs, 
            device=Config.device
        ).flatten()

        preds.append(pred_test)
    
preds = np.median(preds, 0)
pred_test=preds.copy()


# ## Sub

# In[ ]:


df_test['pred'] = pred_test
df_test


# In[ ]:


sub.loc[df_test.index, "pressure"] = pred_test
sub.to_csv(RESULT_PATH+f'{Config.selected_model}_submission_tta5.csv', index=False)


# In[ ]:


sub.shape


# ## SAVE OOF

# In[ ]:


#np.savez_compressed(RESULT_PATH+f'{Config.selected_model}_{metric_value :.3f}.npz',valid=pred_oof, test=pred_test)


# ## POST

# In[ ]:


unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        # If the predicted value is bigger than the highest pressure in the train dataset,
        # return the max value.
        return sorted_pressures[-1]
    elif insert_idx == 0:
        # Same control but for the lower bound.
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val

sub["pressure"] = sub["pressure"].apply(find_nearest)
sub.to_csv(RESULT_PATH+f'{Config.selected_model}_submission_post_tta5.csv', index=False)
sub.head()






