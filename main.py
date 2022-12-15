import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import pickle
import yaml
import datetime
from kmeans_pytorch import kmeans
from utils import *
from networks import *
# from label_generate import *
import argparse

def evaluate(model,data_loader,device,full = True):
    gt = np.array([])
    p = np.array([])
    with torch.no_grad():
        for batch,label in data_loader:
            if full:
                batch = p_normalize(batch).to(device)
            outputs = model(batch,ortho_step=False, mode = 'inference')
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

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_se(senet, train_loader, block,save_path, name,params):
    """
        训练自表达网络
        使用senet的代码块
    """
    epochs = params['epochs']
    batch_size = params['batch_size']
    lr = params['lr']
    save_epoch = params['save_epoch']
    gamma = params['gamma']
    lmbd = params['lmbd']

    optimizer = optim.Adam(senet.parameters(), lr=lr)
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

            k_block = senet.key_embedding(block)
            c = senet.get_coeff(q_batch, k_block)
            rec_batch = rec_batch + c.mm(block)
            reg = reg + regularizer(c, lmbd)
            
            diag_c = senet.thres((q_batch * k_batch).sum(dim=1, keepdim=True)) * senet.shrink
            rec_batch = rec_batch - diag_c * batch
            reg = reg - regularizer(diag_c, lmbd)

            rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
            loss = (0.5 * gamma * rec_loss + reg) / batch_size

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(senet.parameters(), 0.001)
            optimizer.step()

            rec_loss_item += (rec_loss.item() / batch_size)
            reg_loss_item += (reg.item() / batch_size)
            loss_item += loss.item()
        
        scheduler.step()
        
        pbar.set_postfix(loss="{:3.4f}".format(loss_item / n_batch),
                             rec_loss="{:3.4f}".format(rec_loss_item / n_batch),
                             reg="{:3.4f}".format(reg_loss_item / n_batch))
        line = [str(epoch),'%.4f'%(loss_item / n_batch),'%.4f'%(rec_loss_item / n_batch),'%.4f'%(reg_loss_item / n_batch)]
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write('\n'+','.join(map(str, line)))

        if (epoch + 1) % save_epoch == 0:
            torch.save(senet.state_dict(),model_path+'/se_'+name+'_'+str(epoch)+'.pt')

    print("finish training senet!")
    return senet

def write_log(log,path):
    with open(path,'a') as f:
        f.write('\n'+log)
    f.close()
    print(log)

def train(config):
    name = config['dataset']['name']
    data_path = config["dataset"]["path"]
    data_num = config["dataset"]["num"]
    num_cluster = config["dataset"]["num_cluster"]

    if name in ["CIFAR100","CIFAR10","STL10"]:
        full_data = np.load(data_path + "/"+name+"-features.npy")
        full_labels = np.load(data_path + "/"+name+"-labels.npy")

    elif name in ["MNIST", "FashionMNIST", "EMNIST"]:
        with open(data_path+"/"+name+"_scattering_train_data.pkl", 'rb') as f:
            train_samples = pickle.load(f)
        with open(data_path+"/"+name+"_scattering_train_label.pkl", 'rb') as f:
            train_labels = pickle.load(f)
        with open(data_path+"/"+name+"_scattering_test_data.pkl", 'rb') as f:
            test_samples = pickle.load(f)
        with open(data_path+"/"+name+"_scattering_test_label.pkl", 'rb') as f:
            test_labels = pickle.load(f)

        full_data = np.concatenate([train_samples, test_samples], axis=0)
        full_labels = np.concatenate([train_labels, test_labels], axis=0)

    elif name in ["REUTERS"]:
        data = np.load(data_path+'/'+'reutersidf10k.npy',allow_pickle=True).item()
        full_data = data['data']
        full_labels = data['label']

    else:
        raise Exception("The dataset are currently not supported.")    
    device = config['params']['device']
    sampled_idx = np.random.choice(full_data.shape[0], data_num, replace=False)
    data = p_normalize(torch.from_numpy(full_data[sampled_idx]).float()).to(device)
    labels = torch.Tensor(full_labels[sampled_idx])
    train_data = MyDataset(data,labels)
    se_batch_size = config['se_model']['batch_size']
    train_loader = DataLoader(train_data, batch_size=se_batch_size, shuffle=True)

    input_dims, hid_dims, out_dim = config["se_model"]["input_dims"], config[
        "se_model"]["hid_dims"], config["se_model"]["output_dims"]

    senet = SENet(input_dims, hid_dims, out_dim, kaiming_init=True).to(device)
    se_pretrained = config['params']['se_pretrained']
    if se_pretrained:
        se_model_path = config['params']['se_model_path']
        senet.load_state_dict(torch.load(se_model_path))
    else:
        save_path = "se_model/"+name
        params = config["se_model"]
        senet = train_se(senet, train_loader, data,save_path,name,params)
    
    # train spectralnet
    params = config['spec_model']
    params['n_hidden_1'] = params['hid_dims'][0]
    params['n_hidden_2'] = params['hid_dims'][1]
    params ['num_cluster'] = num_cluster

    model = SpectralNet(params).to(device)
    # deepcluster = Kmeans(num_cluster)
    feature,_ = model(data)
    _,cluster_centers = kmeans(X=feature, num_clusters=num_cluster, distance='euclidean', device=torch.device(device))
    model.cluster_layer.data = cluster_centers
    # train params
    lr = params['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.0)
    # criterion = nn.KLDivLoss(reduction = 'batchmean')
    lmda = params['lambda']
    spec_batch_size = params['batch_size']
    
    # epoch params
    epochs = config['spec_model']['epochs']
    spec_save_epoch  = config['spec_model']['save_epoch']
    eval_epoch = config['spec_model']['eval_epoch']

    # path params 
    csv_path = 'spec_model/'+name+'/csv_loss'
    model_path = 'spec_model/'+name+'/spectralnet'
    os.makedirs(csv_path,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)

    # loss log csv file
    csv_file = csv_path+"/spectralnet_" + str(datetime.datetime.now())+".csv"
    headers = ['epoch','loss','loss_sn']#,'loss_ce']
    with open(csv_file, 'w+', encoding='utf-8') as f:
        f.write(','.join(map(str, headers)))

    # ortho dataloader
    train_loader_ortho = DataLoader(train_data, batch_size=spec_batch_size, shuffle=True)
    train_loader_grad = DataLoader(train_data, batch_size=spec_batch_size, shuffle=True)

    print("strat training spectralnet!")
    log = 'dataset: ' + name +'num_sampled_data: '+ str(data_num) + ' batch_size: '+ str(spec_batch_size) + ' learning rate: ' + str(lr)
    path = 'spec_model/'+name+'/evluation_'+name+'.txt'
    write_log(log,path)

    pbar = tqdm(range(epochs), ncols=120)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        # 生成伪标签
        # feature,_ = model(data)
        # cluster,cluster_centers = kmeans(X=feature, num_clusters=num_cluster, distance='euclidean', device=torch.device('cuda'))
        # cluster = cluster.numpy().tolist()
        # # print(cluster)
        # # clustering_loss = deepcluster.cluster(feature.cpu().detach().numpy())
        # # feature = feature.cpu().detach().numpy()
        # # I = deepcluster.cluster(feature)
        
        # train_dataset = cluster_assign(cluster_lists,data)
        # # uniformly sample per target
        # sampler = UnifLabelSampler(int(1 * len(train_dataset)),
        #                            cluster_lists)
        # train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     # num_workers=2,
        #     sampler=sampler,
        #     # pin_memory=True,
        # )
        
        n_batch, loss_sn_item, loss_dec_item, loss_item = 0,0,0,0
        for (data_ortho,_), (data_grad,_) in zip(train_loader_ortho, train_loader_grad):
            n_batch += 1
            # QR分解正交
            x = data_ortho
            x.to(device)

            with torch.no_grad():
                res = model(x, ortho_step=True)

            # 梯度计算
            x = data_grad
            x = x.to(device)

            # compute similarity matrix for the batch
            with torch.no_grad():
                W = torch.abs(senet(x,x))
                W = W + W.T
            
            optimizer.zero_grad()
    
            Y,q = model(x, ortho_step=False)
            tmp_q = q.data
            p = target_distribution(tmp_q)
            Y_dists = (torch.cdist(Y, Y)) ** 2

            loss_sn = (W * Y_dists).mean() * x.shape[0]
            # loss_dec = criterion(q.log(),p)
            loss = loss_sn # + lmda * loss_dec

            loss.backward()
            optimizer.step()

            loss_sn_item += loss_sn.item()
            # loss_dec_item += loss_dec.item()
            loss_item += loss.item()
        
        pbar.set_postfix(loss="{:3.4f}".format(loss_item / n_batch),
                loss_sn="{:3.4f}".format(loss_sn_item / n_batch),)
                # loss_ce="{:3.4f}".format(loss_dec_item / n_batch))

        line = [str(epoch),'%.4f'%(loss_item / n_batch),'%.4f'%(loss_sn_item / n_batch)]#,'%.4f'%(loss_dec_item / n_batch)]
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write('\n'+','.join(map(str, line)))

        scheduler.step()

        if (epoch + 1) % eval_epoch == 0 or (epoch + 1) % eval_epoch == 0:
            feature,_ = model(data)
            _,cluster_centers = kmeans(X=feature, num_clusters=num_cluster, distance='euclidean', device=torch.device(device))
            model.cluster_layer.data = cluster_centers

        if (epoch + 1) % spec_save_epoch == 0:
            torch.save(model.state_dict(),model_path + '/spec_'+name+'_'+str(epoch)+'_'+str(data_num)+'.pt')
        
        if (epoch + 1) % eval_epoch == 0:
            print("Evaluating on sampled data...")
            acc,nmi,pur,ari = evaluate(model,train_loader,device,False)
            log = "sampled dataset in epoch: %d, acc: %.5f, nmi: %.5f, pur: %.5f, ari: %.5f"%(epoch,acc,nmi,pur,ari)
            write_log(log,path)
            
            # 在整个数据集上评估
            full_data_loader = DataLoader(MyDataset(full_data,full_labels), batch_size=spec_batch_size, shuffle=False)
            print("Evaluating on full data...")
            acc,nmi,pur,ari = evaluate(model,full_data_loader,device)
            log = "full dataset in epoch: %d, acc: %.5f, nmi: %.5f, pur: %.5f, ari: %.5f"%(epoch,acc,nmi,pur,ari)
            write_log(log,path)
        
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MNIST")
    args = parser.parse_args()

    assert args.dataset in ['CIFAR10','CIFAR100','STL10','MNIST','FashionMNIST','EMNIST','REUTERS'], 'The dataset are currently not supported.'
    config_file = open("./config/{}.yaml".format(args.dataset),'r')
    config = yaml.load(config_file,Loader=yaml.FullLoader)
    same_seeds(1)
    train(config)