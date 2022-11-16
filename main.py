import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip
import yaml
import datetime
from utils import *
from networks import *
from label_generate import *

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def train_se(senet, train_loader,batch_size, epochs,save_epoch):
    """
        训练自表达网络
        使用senet的代码块
    """
    optimizer = optim.Adam(senet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    os.makedirs("se_model",exist_ok=True)
    csv_file = "se_model/senet_" + str(datetime.datetime.now())+".csv"
    headers = ['epoch','loss','loss_rec','loss_reg']
    with open(csv_file, 'w+', encoding='utf-8') as f:
        f.write(','.join(map(str, headers)))
    print("strat training senet!")
    pbar = tqdm(range(epochs), ncols=120)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        n_batch ,rec_loss_item ,reg_loss_item ,loss_item = 0, 0, 0, 0
        for batch, _ in train_loader:
            n_batch += 1

            q_batch = senet.query_embedding(batch)
            k_batch = senet.key_embedding(batch)
            # 损失计算
            rec_batch = torch.zeros_like(batch)
            reg = torch.zeros([1]).cuda()
            for block,_ in train_loader:
                k_block = senet.key_embedding(block)
                c = senet.get_coeff(q_batch, k_block)
                rec_batch = rec_batch + c.mm(block)
                reg = reg + regularizer(c, 0.9)
            
            diag_c = senet.thres((q_batch * k_batch).sum(dim=1, keepdim=True)) * senet.shrink
            rec_batch = rec_batch - diag_c * batch
            reg = reg - regularizer(diag_c, 0.9)

            rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
            loss = (0.5 * 200 * rec_loss + reg) / batch_size

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(senet.parameters(), 0.001)
            optimizer.step()

            rec_loss_item += (rec_loss.item() / batch_size)
            reg_loss_item += (reg.item() / batch_size)
            loss_item += loss.item()
        
        pbar.set_postfix(loss="{:3.4f}".format(loss_item / n_batch),
                             rec_loss="{:3.4f}".format(rec_loss_item / n_batch),
                             reg="{:3.4f}".format(reg_loss_item / n_batch))
        line = [str(epoch),'%.4f'%(loss_item / n_batch),'%.4f'%(rec_loss_item / n_batch),'%.4f'%(reg_loss_item / n_batch)]
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write('\n'+','.join(map(str, line)))
        scheduler.step()

        if (epoch + 1) % save_epoch == 0:
            torch.save(senet.state_dict(),'se_model/se_model_'+str(epoch)+'.pt')

    print("finish training senet!")
    return senet

def train(config):
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["params"]["batch_size"]
    train_data = MyDataset(data_path=data_path, data_num=data_num)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim,se_epochs,se_save_epoch = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"], config['se_model']['epochs'], config['se_model']['save_epochs']

    device = config['device']
    senet = SENet(input_dims, hid_dims, out_dim).to(device)
    # senet = train_se(senet, train_loader,batch_size, se_epochs,se_save_epoch)

    train_loader_ortho = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    num_cluster = config['spec_model']['num_cluster']
    params = {'input_dims':config['spec_model']['input_dims'],'num_cluster':num_cluster,'n_hidden_1':config['spec_model']['hid_dims'][0],
        'n_hidden_2':config['spec_model']['hid_dims'][1],'epsilon':config['spec_model']['epsilon'],}
    model = SpectralNet(params).to(device)
    learning_rate = config['params']['lr']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    deepcluster = Kmeans(num_cluster)
    criterion = nn.CrossEntropyLoss()
    
    epochs = config['spec_model']['epochs']
    spec_save_epoch  = config['spec_model']['save_epoch']
    os.makedirs("spec_model",exist_ok=True)
    csv_file = "spec_model/spectralnet_" + str(datetime.datetime.now())+".csv"
    headers = ['epoch','loss','loss_sn','loss_ce']
    with open(csv_file, 'w+', encoding='utf-8') as f:
        f.write(','.join(map(str, headers)))

    print("strat training spectralnet!")
    pbar = tqdm(range(epochs), ncols=120)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        # 生成伪标签
        feature,_ = model(train_data.data)
        clustering_loss = deepcluster.cluster(feature.cpu().data.numpy())
        train_dataset = cluster_assign(deepcluster.cluster_lists,train_data.data)
        # uniformly sample per target
        sampler = UnifLabelSampler(int(1 * len(train_dataset)),
                                   deepcluster.cluster_lists)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # num_workers=2,
            sampler=sampler,
            # pin_memory=True,
        )
        n_batch, loss_sn_item , loss_ce_item,loss_item = 0,0,0,0

        for (data_ortho,_), (data_grad,label) in zip(train_loader_ortho, train_dataloader):
            n_batch += 1
            # QR分解正交
            x = data_ortho
            x.to(device)

            with torch.no_grad():
                res = model(x, ortho_step=True)

            # 梯度计算
            x, target = data_grad,label
            x, target = x.to(device), target.to(device)

            # compute similarity matrix for the batch
            with torch.no_grad():
                W = torch.abs(senet.get_coeff(x,x))
            
            optimizer.zero_grad()

            Y,P = model(x, ortho_step=False)
            Y_dists = (torch.cdist(Y, Y)) ** 2
            loss_sn = (W * Y_dists).mean() * x.shape[0]

            loss_ce = criterion(P,target)
            loss = (loss_sn + 100 * loss_ce) / batch_size

            loss.backward()
            # optimizer.step()

            loss_sn_item += (loss_sn.item() / batch_size)
            loss_ce_item += (loss_ce.item() / batch_size)
            loss_item += loss.item()
        
        pbar.set_postfix(loss="{:3.4f}".format(loss_item / n_batch),
                loss_sn="{:3.4f}".format(loss_sn_item / n_batch),
                loss_ce="{:3.4f}".format(loss_ce_item / n_batch))

        line = [str(epoch),'%.4f'%(loss_item / n_batch),'%.4f'%(loss_sn_item / n_batch),'%.4f'%(loss_ce_item / n_batch)]
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write('\n'+','.join(map(str, line)))

        scheduler.step()
        if (epoch + 1) % spec_save_epoch == 0:
            torch.save(senet.state_dict(),'spec_model/spec_model_'+str(epoch)+'.pt')
        
    print("finish training spectralnet!")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    config_file = open("./config/config_init.yaml",'r')
    config = yaml.load(config_file,Loader=yaml.FullLoader)
    same_seeds(1)
    train(config)