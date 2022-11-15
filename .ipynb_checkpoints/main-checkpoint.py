import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip
import yaml
from utils import *
from networks import *
from label_generate import *

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def train_se(senet, train_loader,batch_size, epochs):
    """
        训练自表达网络
        使用senet的代码块
    """
    optimizer = optim.Adam(senet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    print("strat training senet!")
    for epoch in range(epochs):
        print("epoch:", epoch)
        pbar = tqdm(train_loader)
        for batch, _ in pbar:
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
            # print("reg loss:",reg.item(),"rec loss:",rec_loss.item())
            loss = (0.5 * 200 * rec_loss + reg) / batch_size

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(senet.parameters(), 0.001)
            optimizer.step()
        
        scheduler.step()
        """
        TODO
        Loss记录
        模型保存，评估
        """
    os.makedirs("se_model",exist_ok=True)
    torch.save(senet.state_dict(),'se_model/se_model.pt')
    print("finish training senet!")
    return senet

def train(config):
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["params"]["batch_size"]
    train_data = MyDataset(data_path=data_path, data_num=data_num)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim,se_epochs = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"], config['se_model']['epochs']

    device = config['device']
    senet = SENet(input_dims, hid_dims, out_dim).to(device)
    # senet = train_se(senet, train_loader,batch_size, se_epochs)

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
    print("strat training spectralnet!")
    for epoch in range(epochs):
        print("epoch:", epoch)
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

        for (data_ortho,_), (data_grad,label) in tzip(train_loader_ortho, train_dataloader):
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
            print("spec loss:",loss_sn.item(),"cross entropy loss:",loss_ce.item())
            loss = loss_sn + loss_ce
            loss.backward()
        
        scheduler.step()
        """
        TODO
        Loss记录
        模型保存，评估
        """
    torch.save(senet.state_dict(),'spec_model/spec_model.pt')
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

