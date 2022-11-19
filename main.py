import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import yaml
import datetime
from utils import *
from networks import *
from label_generate import *

def evaluate(model,data_loader,full = True):
    gt = np.array([])
    p = np.array([])
    with torch.no_grad():
        for batch,label in data_loader:
            if full:
                batch = p_normalize(batch).cuda()
            _,outputs = model(batch)
            _,pred = torch.max(outputs, 1)
            p = np.append(p,pred.cpu().detach().numpy())
            gt = np.append(gt,label.cpu().detach().numpy())
    
    acc = Accuracy(p,gt)
    nmi = normalized_mutual_info_score(p,gt)
    pur = purity(p,gt)
    ari = adjusted_rand_score(p,gt)

    return acc,nmi,pur,ari

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)

def train_se(senet, train_loader, block, batch_size, epochs,save_epoch,save_path,name):
    """
        训练自表达网络
        使用senet的代码块
    """
    optimizer = optim.Adam(senet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    os.makedirs("se_model",exist_ok=True)
    csv_path = save_path+'/csv_loss'
    model_path = save_path+'/senet'
    os.makedirs(csv_path,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)
    csv_file = csv_path + "/senet_" + str(datetime.datetime.now())+".csv"
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
            # for block,_ in train_loader:
            #     k_block = senet.key_embedding(block)
            #     c = senet.get_coeff(q_batch, k_block)
            #     rec_batch = rec_batch + c.mm(block)
            #     reg = reg + regularizer(c, 0.9)
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
            torch.save(senet.state_dict(),model_path+'/se_'+name+'_'+str(epoch)+'.pt')

    print("finish training senet!")
    return senet

def train(config):
    name = config['dataset']['name']
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    batch_size = config["params"]["batch_size"]

    if name == "CIFAR100":
        full_data = np.load(data_path + "/cifar100_features.npy")
        full_labels = np.load(data_path + "/cifar100_labels.npy")
    elif name == "CIFAR10":
        full_data = np.load(data_path + "/cifar10_features.npy")
        full_labels = np.load(data_path + "/cifar10_labels.npy")
    elif name == "MNIST":
        full_data = np.load(data_path + "/cifar100_features.npy")
        full_labels = np.load(data_path + "/cifar100_labels.npy")
    elif name == "FashionMNIST":
        full_data = np.load(data_path + "/cifar100_features.npy")
        full_labels = np.load(data_path + "/cifar100_labels.npy")
    elif name == "EMNIST":
        full_data = np.load(data_path + "/cifar100_features.npy")
        full_labels = np.load(data_path + "/cifar100_labels.npy")
    elif name == "STL10":
        full_data = np.load(data_path + "/stl10_features.npy")
        full_labels = np.load(data_path + "/stl10_labels.npy")
    elif name == "REUTERS":
        full_data = np.load(data_path + "/cifar100_features.npy")
        full_labels = np.load(data_path + "/cifar100_labels.npy")
    else:
        raise Exception("The dataset are currently not supported.")    

    sampled_idx = np.random.choice(full_data.shape[0], data_num, replace=False)
    data = p_normalize(torch.from_numpy(full_data[sampled_idx]).float()).cuda()
    labels = torch.Tensor(full_labels[sampled_idx])
    train_data = MyDataset(data,labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    input_dims, hid_dims, out_dim,se_epochs,se_save_epoch = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"], config['se_model']['epochs'], config['se_model']['save_epoch']

    device = config['device']
    senet = SENet(input_dims, hid_dims, out_dim).to(device)
    save_path = "se_model/"+name
    senet = train_se(senet, train_loader, data, batch_size, se_epochs,se_save_epoch,save_path,name)

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
    eval_epoch = config['spec_model']['eval_epoch']

    csv_path = 'spec_model/'+name+'/csv_loss'
    model_path = 'spec_model/'+name+'/spectralnet'
    os.makedirs(csv_path,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)

    csv_file = csv_path+"/spectralnet_" + str(datetime.datetime.now())+".csv"
    headers = ['epoch','loss','loss_sn','loss_ce']
    with open(csv_file, 'w+', encoding='utf-8') as f:
        f.write(','.join(map(str, headers)))

    print("strat training spectralnet!")
    pbar = tqdm(range(epochs), ncols=120)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        # 生成伪标签
        feature,_ = model(data)
        clustering_loss = deepcluster.cluster(feature.cpu().detach().numpy())
        train_dataset = cluster_assign(deepcluster.cluster_lists,data)
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
            optimizer.step()

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
            torch.save(model.state_dict(),model_path + '/spec_'+name+'_'+str(epoch)+'.pt')
        
        if (epoch + 1) % eval_epoch == 0:
            print("Evaluating on sampled data...")
            acc,nmi,pur,ari = evaluate(model,train_loader,False)
            log = "epoch:"+str(epoch)+" acc:"+str(acc)+"nmi:"+str(nmi)+"pur:"+str(pur)+"ari:"+str(ari)
            with open('spec_model/'+name+'/evluation_'+name+'.txt','a') as f:
                f.write('\n'+log)
            f.close()
            print(log)
        
    print("finish training spectralnet!")

    # 在整个数据集上评估
    full_data_loader = DataLoader(MyDataset(full_data,full_labels), batch_size=batch_size, shuffle=False)
    print("Evaluating on full data...")
    acc,nmi,pur,ari = evaluate(model,full_data_loader)
    log = "full data: acc:"+str(acc)+"nmi:"+str(nmi)+"pur:"+str(pur)+"ari:"+str(ari)
    with open('spec_model/'+name+'/evluation_'+name+'.txt','a') as f:
        f.write('\n'+log)
    f.close()
    print(log)

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