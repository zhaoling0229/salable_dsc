import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from networks import *
from tqdm import tqdm
from tqdm.contrib import tzip
import yaml

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def train_se(senet, train_loader,batch_size, epochs):
    """
        训练自表达网络
        使用senet的代码块
    """
    optimizer = optim.Adam(senet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
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


def train(config,params):

    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["dataset"]["batch_size"]
    data = MyDataset(data_path=data_path, data_num=data_num)
    # train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"]

    senet = SENet(input_dims, hid_dims, out_dim)
    # train_se(senet, train_loader, 100000)

    train_loader_grad = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)
    train_loader_ortho = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)
    model = SpectralNet(params)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(10000):
        print("epoch:", epoch)
        for data_ortho, data_grad in tzip(train_loader_ortho, train_loader_grad):
            # orthogonalization step
            x, target = data_ortho
            x, target = x.to(device), target.to(device)
            # x = x.view(x.shape[0], input_sz)

            with torch.no_grad():
                res = model(x, ortho_step=True)

            # gradient step
            x, target = data_grad
            x, target = x.to(device), target.to(device)
            # x = x.view(x.shape[0], input_sz)

            # compute similarity matrix for the batch
            with torch.no_grad():
                W = torch.abs(senet.get_coeff(x,x))
            
            optimizer.zero_grad()

            Y,P = model(x, ortho_step=False)
            Y_dists = (torch.cdist(Y, Y)) ** 2
            loss_sn = (W * Y_dists).mean() * x.shape[0]

            loss = loss_sn
            print("spec loss:",loss.item())
            loss.backward()
        
        scheduler.step()

if __name__ == "__main__":
    """
    主函数中把所有逻辑串起来
    1. 加载数据 
    2. 实例化model
    3. 加载配置文件
    4. 训练
    5. 评估模型
    6. 记录结果
    """
    # config_path = ""
    # config = read_config(config_path)
    # data = np.load("E:/dataset/feature/CIFAR100-MCR2/cifar100_features.npy")
    # label = np.load("E:/dataset/feature/CIFAR100-MCR2/cifar100_labels.npy")
    # mydataset = MyDataset("E:/dataset/feature/CIFAR100-MCR2", 1500)
    # train_loader = DataLoader(mydataset, batch_size=200, shuffle=False)
    # senet = SENet(128, [1024,1024], 1024)
    # train_se(senet,train_loader,200,1000)
    params = {
        'k': 5,
        "n_hidden_1": 1024,
        "n_hidden_2": 512,
        "batch_size": 100,
        "gamma": 23,
        'epsilon': 1e-7,
        "input_sz": 128,
        "affinity": "rbf",
        'lr': 0.001,
        'n_epochs': 5000,
        'save_every': 50,
        'print_every': 15,
        'log_every': 5,
        'path': "",
        'dataset': "cc",
        "stop_acc": 0.997,
        'to_wandb': False,
        'device': 'cpu'
    }
    # model = SpectralNet(params)
    # for epoch in range(10000):
    #     print("epoch:", epoch)
    #     pbar = tqdm(train_loader)
    #     for feature, _ in pbar:
    #         Y, P = model(feature)
    #         print(Y)
    config_file = open("E:/PythonProject/scalable_dsc/config/config_init.yaml",'r')
    config = yaml.load(config_file)
    # print(config)
    train(config,params)

