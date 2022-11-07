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
            reg = torch.zeros([1])
            for block,_ in train_loader:
                k_block = senet.key_embedding(block)
                c = senet.get_coeff(q_batch, k_block)
                rec_batch = rec_batch + c.mm(block)
                reg = reg + regularizer(c, 0.9)
            
            diag_c = senet.thres((q_batch * k_batch).sum(dim=1, keepdim=True)) * senet.shrink
            rec_batch = rec_batch - diag_c * batch
            reg = reg - regularizer(diag_c, 0.9)

            rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
            print("reg loss:",reg.item(),"rec loss:",rec_loss.item())
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
    torch.save(senet.state_dict(),'se_model/se_model.pt')
    print("finish training senet!")
    return senet

def train(config):
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["dataset"]["batch_size"]
    data = MyDataset(data_path=data_path, data_num=data_num)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"]

    senet = SENet(input_dims, hid_dims, out_dim)
    senet = train_se(senet, train_loader, 100000)

    train_loader_ortho = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)
    
    num_cluster = config['spec_model']['num_cluster']
    params = {'input_dims':config['spec_model']['input_dims'],'num_cluster':num_cluster,'n_hidden_1':config['spec_model']['hid_dims'][0],
        'n_hidden_1':config['spec_model']['hid_dims'][1],'epsilon':config['spec_model']['epsilon'],}
    model = SpectralNet(params)
    learning_rate = config['params']['lr']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    device = config['device']
    deepcluster = Kmeans(num_cluster)
    criterion = nn.CrossEntropyLoss()
    
    print("strat training spectralnet!")
    for epoch in range(10000):
        print("epoch:", epoch)
        # 生成伪标签
        feature,_ = model(data)
        clustering_loss = deepcluster.cluster(feature)
        train_dataset = cluster_assign(deepcluster.images_lists,data)
        # uniformly sample per target
        sampler = UnifLabelSampler(int(1 * len(train_dataset)),
                                   deepcluster.images_lists)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=2,
            sampler=sampler,
            pin_memory=True,
        )

        for (data_ortho,_), (data_grad,label) in tzip(train_loader_ortho, train_dataloader):
            # QR分解正交
            x, target = data_ortho
            x, target = x.to(device), target.to(device)

            with torch.no_grad():
                res = model(x, ortho_step=True)

            # 梯度计算
            x, target = data_grad
            x, target = x.to(device), target.to(device)

            # compute similarity matrix for the batch
            with torch.no_grad():
                W = torch.abs(senet.get_coeff(x,x))
            
            optimizer.zero_grad()

            Y,P = model(x, ortho_step=False)
            Y_dists = (torch.cdist(Y, Y)) ** 2
            loss_sn = (W * Y_dists).mean() * x.shape[0]
            # CELoss
            loss_ce = criterion(P,label)
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

if __name__ == "__main__":
    config_file = open("E:/PythonProject/scalable_dsc/config/config_init.yaml",'r')
    config = yaml.load(config_file)
    train(config)

