from src.model import *
from torch import optim
from tqdm import tqdm, trange
from src.pytorchtools import EarlyStopping
from src.utils import optimal_threshold, f1, pre_rec_f1
import torch
import random
import logging
from tqdm import tqdm
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader
from sklearn.metrics import average_precision_score


def build_dataloader_topic(data, mask, target, batch_size, mode="test"):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float), torch.tensor(mask, dtype=torch.float), torch.tensor(target, dtype=torch.float))
    if mode == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=int(batch_size), sampler=sampler)
    return dataloader

def build_dataloader(data, target, batch_size, mode="test"):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.float))
    if mode == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=int(batch_size), sampler=sampler)
    return dataloader



def train_topic(config, num,model_fname, train_data, valid_data, train_label, valid_label, train_mask, valid_mask):
    logging.info("build data loader")
    train_loader = build_dataloader_topic(train_data, train_mask, train_label, config.batch_size, "train")
    dev_loader = build_dataloader_topic(valid_data, valid_mask, valid_label, config.batch_size)
    model = Binary_topic_Net(config, num)

    optimizer = optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, eps=1e-10)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)  # random batch training seed (shuffle) to ensure reproducibility

    early_stopping = EarlyStopping(patience=40, verbose=True, path=model_fname)

    if config.gpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            torch.cuda.manual_seed_all(config.seed)
        model.to('cuda')

    model.train()
    logging.info("start training...")
    for epoch in trange(int(config.num_epoch), desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            X,m,y = batch
            if config.gpu:
                X = X.to('cuda')
                y = y.to('cuda')
                m = m.to('cuda')
            optimizer.zero_grad()

            # forward
            _, loss = model(X, m, y)
            loss = loss.mean()

            # backward
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss = tr_loss / (step + 1)
        # validation
        model.eval()
        ys_dev = []
        ps_dev = []
        loss_valid = 0
        for step, batch_dev in enumerate(tqdm(dev_loader, desc="Validation")):
            X_val, m_val, y_val = batch_dev
            if config.gpu:
                X_val = X_val.to('cuda')
                m_val = m_val.to('cuda')
                y_val = y_val.to('cuda')
            prob, bt_loss = model(X_val,m_val,y_val)
            loss_valid += bt_loss.mean()
            ps_dev.extend(prob.data.cpu().clone().numpy())
            ys_dev.extend(y_val.data.cpu().clone().numpy())
        loss_valid = loss_valid / (step + 1)

        th, dev_f1 = optimal_threshold(ys_dev, ps_dev, Ns=50)

        logging.info("Epoch: %d | train loss: %.4f | valid loss: %.4f | valid f1: %.4f ",
                     epoch + 1, tr_loss, loss_valid, dev_f1)

        model.train()

        early_stopping(-dev_f1, model)
        # lr_scheduler.step(loss_valid)
        if early_stopping.early_stop:
            logging.info("Early Stopping. Model trained.")
            break


def train(config, model_fname, train_data, valid_data, train_label, valid_label):
    logging.info("build data loader")
    train_loader = build_dataloader(train_data, train_label, config.batch_size, "train")
    dev_loader = build_dataloader(valid_data, valid_label, config.batch_size)
    model = Classifier(config)

    optimizer = optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, eps=1e-10)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)  # random batch training seed (shuffle) to ensure reproducibility

    early_stopping = EarlyStopping(patience=40, verbose=True, path=model_fname)

    if config.gpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            torch.cuda.manual_seed_all(config.seed)
        model.to('cuda')

    model.train()
    logging.info("start training...")
    for epoch in trange(int(config.num_epoch), desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            X,y = batch
            if config.gpu:
                X = X.to('cuda')
                y = y.to('cuda')
            optimizer.zero_grad()

            # forward
            _, loss = model(X, y)
            loss = loss.mean()

            # backward
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss = tr_loss / (step + 1)
        # validation
        model.eval()
        ys_dev = []
        ps_dev = []
        loss_valid = 0
        for step, batch_dev in enumerate(tqdm(dev_loader, desc="Validation")):
            X_val, y_val = batch_dev
            if config.gpu:
                X_val = X_val.to('cuda')
                y_val = y_val.to('cuda')
            prob, bt_loss = model(X_val,y_val)
            loss_valid += bt_loss.mean()
            ps_dev.extend(prob.data.cpu().clone().numpy())
            ys_dev.extend(y_val.data.cpu().clone().numpy())
        loss_valid = loss_valid / (step + 1)

        th, dev_f1 = optimal_threshold(ys_dev, ps_dev, Ns=50)

        logging.info("Epoch: %d | train loss: %.4f | valid loss: %.4f | valid f1: %.4f ",
                     epoch + 1, tr_loss, loss_valid, dev_f1)

        model.train()

        early_stopping(-dev_f1, model)
        # lr_scheduler.step(loss_valid)
        if early_stopping.early_stop:
            logging.info("Early Stopping. Model trained.")
            break



def test_topic(config,test_data,m,test_target,model=None, th=None, eval_mark='test'):

    test_loader = build_dataloader_topic(test_data,m,test_target,config.batch_size)

    if config.gpu:
        if torch.cuda.device_count() > 1:
            logging.info('use multiple gpu')
            model = torch.nn.DataParallel(model)
        torch.cuda.manual_seed_all(config.seed)
        model.to('cuda')

    model.eval()
    logging.info("prediction")
    total_label = []
    total_probs = []
    for batch in tqdm(test_loader):
        X, m, y = batch
        if config.gpu:
            X = X.to('cuda')
            y = y.to('cuda')
            m = m.to('cuda')
        prob = model(X,m)

        total_label.extend(y.data.cpu().clone().numpy())
        total_probs.extend(prob.data.cpu().clone().numpy())

    total_probs = np.array(total_probs)
    total_label = np.array(total_label)
    if 'dev' in eval_mark and th == None:
        best_th, best_f1 = optimal_threshold(total_label, total_probs, Ns=1000)
        return best_th

    ap = round(average_precision_score(total_label, total_probs), 4)

    pre_label = np.zeros(total_label.shape, dtype=int)
    idx = np.where(total_probs >= th)[0]
    pre_label[idx] = 1
    pre, rec, f1 = pre_rec_f1(total_label, pre_label)
    return [ap, pre, rec, f1]

def test(config,test_data,test_target,model=None, th=None, eval_mark='test'):

    test_loader = build_dataloader(test_data,test_target,config.batch_size)

    if config.gpu:
        if torch.cuda.device_count() > 1:
            logging.info('use multiple gpu')
            model = torch.nn.DataParallel(model)
        torch.cuda.manual_seed_all(config.seed)
        model.to('cuda')

    model.eval()
    logging.info("prediction")
    total_label = []
    total_probs = []
    for batch in tqdm(test_loader):
        X, y = batch
        if config.gpu:
            X = X.to('cuda')
            y = y.to('cuda')
        prob = model(X)

        total_label.extend(y.data.cpu().clone().numpy())
        total_probs.extend(prob.data.cpu().clone().numpy())

    total_probs = np.array(total_probs)
    total_label = np.array(total_label)
    if 'dev' in eval_mark and th == None:
        best_th, best_f1 = optimal_threshold(total_label, total_probs, Ns=1000)
        return best_th

    ap = round(average_precision_score(total_label, total_probs), 4)

    pre_label = np.zeros(total_label.shape, dtype=int)
    idx = np.where(total_probs >= th)[0]
    pre_label[idx] = 1
    pre, rec, f1 = pre_rec_f1(total_label, pre_label)
    return [ap, pre, rec, f1]