import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
from precess import load_data_PeNGaRoo
from torch.utils import data
from model.model import BiRNN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,average_precision_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.use_deterministic_algorithms(True)

batch_size = 16
device = torch.device('cuda')





def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=True)


def xavier_init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


if __name__ == '__main__':
    
    acc = 0
    f1 = 0
    pre = 0
    recall = 0
    auc_score = 0
    mcc = 0
    spe = 0
    aurpc = 0
    ks = 10
    kf = StratifiedKFold(n_splits=ks, random_state=4, shuffle=True)
    X,Xs, y, test_X,tests, test_y, vocab = load_data_PeNGaRoo(1000)
    for index, (train, test) in enumerate(kf.split(X, y)):

        net = BiRNN(21, 32, 512, 2, 0.5).to(device)
        net.apply(xavier_init_weights)

        lr = 0.00005
        weight_decay = 5e-3
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90], 0.9)
        criterion = nn.CrossEntropyLoss()
        X_train = X[train]
        X_trains = Xs[train]
        y_train = y[train]
        X_test = X[test]
        X_tests = Xs[test]
        y_test = y[test]

        train_iter = load_array((torch.tensor(X_train, dtype=torch.long), torch.tensor(X_trains, dtype=torch.float32),torch.tensor(y_train, dtype=torch.long)),
                                batch_size, True)
        # test_iter = load_array((torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)),
        #                        batch_size)
        score_best = 0.0
        acc_best = 0.0

        pre_best = 0.0
        recall_best = 0.0
        auc_best = 0.0
        f1_best = 0.0
        spe_best = 0.0
        AUPR_best = 0.0
        print(f'cross_validation: {index + 1}')
        net.train()
        for epoch in range(150):
            # scheduler.step()
            # print(f'epoch: {epoch + 1}')
            tfakes = []
            tfakes_p = []
            tyts = []
            net.train()
            for x_data,xs, y_data in train_iter:


                x_data = x_data.to(device)
                xs = xs.to(device)
                y_data = y_data.to(device)
                fake = net(x_data,xs.unsqueeze(1))
                tfakes += torch.argmax(fake.cpu().detach(), dim=1).numpy().tolist()
                # print(fakes)
                tfakes_p += fake[:, 1].cpu().detach().numpy().tolist()
                tyts += y_data.cpu().detach().view(-1).numpy().tolist()
                # c_loss.backward()
                loss = criterion(fake, y_data)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(net.parameters(), 0.5)
                optimizer.step()

            # print(f'train acc: {accuracy_score(tyts, tfakes)}, pre: {precision_score(tyts, tfakes)},'
            #       f' recall: {recall_score(tyts, tfakes)}, f1：{f1_score(tyts, tfakes)}, MCC: {matthews_corrcoef(tyts, tfakes)}, AUC: {roc_auc_score(tyts, tfakes_p)}')
            net.eval()

            yts = y_test

            t_X = torch.tensor(X_test, dtype=torch.long).to(device)
            ts = torch.tensor(X_tests,dtype=torch.float32).to(device)
            fake_test = net(t_X,ts.unsqueeze(1)).cpu().detach()

            fakes = torch.argmax(fake_test, dim=1).numpy().tolist()
            # print(fakes)
            fakes_p = fake_test[:, 1].numpy().tolist()
            con = confusion_matrix(yts, fakes)
            TP = con[1, 1]
            TN = con[0, 0]
            FP = con[0, 1]
            FN = con[1, 0]
            spe_ = TN / float(TN + FP)
            score_ = matthews_corrcoef(yts, fakes)
            acc_ = accuracy_score(yts, fakes)
            f1_ = f1_score(yts, fakes)
            pre_ = precision_score(yts, fakes)
            recall_ = recall_score(yts, fakes)
            auc_ = roc_auc_score(yts, fakes_p)
            aupr_ = average_precision_score(yts, fakes_p)
            # print(f'test acc:{acc_}, pre: {pre_}, recall: {recall_}, f1: {f1_}, MCC: {score_}, AUC: {auc_}')
            if score_ > score_best:
                score_best = score_
                acc_best = acc_
                pre_best = pre_
                recall_best = recall_
                f1_best = f1_
                auc_best = auc_
                spe_best = spe_
                AUPR_best = aupr_

        acc += acc_best
        spe += spe_best
        aurpc += AUPR_best
        f1 += f1_best
        pre += pre_best
        recall += recall_best
        auc_score += auc_best
        mcc += score_best
        print('recall: ', recall_best)
        print('spe: ', spe_best)

        print('pre: ', pre_best)
        print('acc: ', acc_best)
        print('f1: ', f1_best)
        print('mcc: ', score_best)
        print('auc: ', auc_best)
        print('aurpc: ', AUPR_best)

    print('recall: ', recall / 10)
    print('spe: ', spe / 10)

    print('pre: ', pre/10)
    print('acc: ', acc / 10)
    print('f1: ', f1/10)
    print('mcc: ', mcc/10)
    print('auc: ', auc_score/10)
    print('aurpc: ', aurpc/10)
