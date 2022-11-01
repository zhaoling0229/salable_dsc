import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from networks import *
from tqdm import tqdm
import yaml

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def train_se(senet, train_loader,batch_size, epochs):
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


def train(config):
    """
    训练的脚本，待实现的逻辑为：
    1. 传入config配置文件以
    2. 加载数据，采用dataloader或者其他
    3. 数据输入model，得到输出
    4. 根据输出进行聚类，生成伪标签
    5. 计算相似度，选出高的作为正样本，低的作为负样本
    6. 计算损失函数，包括谱聚类损失，伪监督损失以及对比损失
    7. 反向传播，记录损失
    8. 保存模型
    """
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["dataset"]["batch_size"]
    data = MyDataset(data_path=data_path, data_num=data_num)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"]

    senet = SENet(input_dims, hid_dims, out_dim)
    train_se(senet, train_loader, 100000)

    model = SpectralNet(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer)
    for epoch in range(10000):
        print("epoch:", epoch)
        pbar = tqdm(train_loader)
        for batch, _ in pbar:
            out = model(batch)
            # loss = compute_spec_loss(out, batch)
            # loss.backward()
            # optimizer.step()
            # scheduler.step()


def read_config(path):
    """
    解析配置文件，输入为配置文件路径，输出为字典
    """
    pass


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
    mydataset = MyDataset("E:/dataset/feature/CIFAR100-MCR2", 1500)
    train_loader = DataLoader(mydataset, batch_size=200, shuffle=False)
    senet = SENet(128, [1024,1024], 1024)
    train_se(senet,train_loader,200,1000)
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
    # config_file = open("E:/PythonProject/scalable_dsc/config/config_init.yaml",'r')
    # config = yaml.load(config_file)
    # print(config)
