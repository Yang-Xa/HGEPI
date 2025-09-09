import numpy as np
from dataset_chr import *
from model import HGEPI
from sklearn.metrics import auc
import torch
import torch.nn as nn
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
EPG_data = EPG_data.to(device)

test_auc = 0.0
test_aupr = 0.0

points_to_evaluate = [

]

fpr = []
tpr = []
precision = []
recall = []
AUC_TEST_LIST = []
num2 = 0


def train(params):
    global test_auc, test_aupr, tpr, fpr, num2
    lr = params['learning_rate1']
    wd = params['weight_decay1']
    dp = params['dropout_rate1']
    hc = params['hidden_channels1']
    head = params['heads1']
    dp2 = params['dropout_rate2']
    head2 = params['heads2']
    outc = params['out_channels1']
    train_loss, train_auc, train_precision = [], [], []
    val_loss, val_auc, val_precision = [], [], []

    model = HGEPI(dp, dp2, head, head2, hc, outc)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(200):
        t_loss = []
        for train_edge_index, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            pred_han = model(train_edge_index, EPG_data)
            loss1 = model.recon_loss(pred_han, labels.to(device))
            loss1.backward()
            optimizer.step()

            t_loss.append(loss1.item())

        train_loss.append(sum(t_loss) / len(t_loss))

        v_y = []
        v_pre = []
        with torch.no_grad():
            for val_edge_index, labels in val_loader:
                model.eval()
                pred_han = model(val_edge_index, EPG_data)
                a, b = model.test_recon_loss(pred_han, labels.to(device))
                v_y.append(a)
                v_pre.append(b)
            v_y = np.concatenate(v_y, axis=0)
            v_pre = np.concatenate(v_pre, axis=0)
            v_fpr, v_tpr, _ = roc_curve(v_y, v_pre)
            AUC_TEST_LIST.append(auc(v_fpr, v_tpr))

    # test
    t_y = []
    t_pre = []
    with torch.no_grad():
        for test_edge_index, labels in test_loader:
            model.eval()
            pred_han = model(test_edge_index, EPG_data)
            a, b = model.test_recon_loss(pred_han, labels.to(device))
            t_y.append(a)
            t_pre.append(b)
    t_y = np.concatenate(t_y, axis=0)
    t_pre = np.concatenate(t_pre, axis=0)
    fpr, tpr, threshold = roc_curve(t_y, t_pre)
    precision, recall, thresholds = precision_recall_curve(t_y, t_pre)

    torch.save(fpr, '/home/yx/rasha/people_gene/shoot_code/chr_res/kz/fpr_%d.pkl' % num2)
    torch.save(tpr, '/home/yx/rasha/people_gene/shoot_code/chr_res/kz/tpr_%d.pkl' % num2)
    torch.save(precision, '/home/yx/rasha/people_gene/shoot_code/chr_res/kz/precision_%d.pkl' % num2)
    torch.save(recall, '/home/yx/rasha/people_gene/shoot_code/chr_res/kz/recall_%d.pkl' % num2)
    num2 = num2 + 1
    torch.save(model, '/home/yx/rasha/people_gene/shoot_code/chr_res/kz/shoot_%d.pth' % num2)
    return -auc(fpr, tpr)

space = {
    'learning_rate1': hp.loguniform('learning_rate1', -11, -2),
    'weight_decay1': hp.loguniform('weight_decay1', -11, -6),
    'dropout_rate1': hp.choice('dropout_rate1', [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]),
    'dropout_rate2': hp.choice('dropout_rate2', [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]),
    'heads1': hp.choice('heads1', [4, 8, 16]),
    'heads2': hp.choice('heads2', [4, 8, 16]),
    'hidden_channels1': hp.choice('hidden_channels1', [1024, 512, 256]),
    'out_channels1': hp.choice('out_channels1', [512, 256, 128])
}
trials = Trials()

best = fmin(fn=train, space=space, algo=tpe.suggest, max_evals=40, points_to_evaluate=points_to_evaluate,
            trials=trials)

l = 0
for i in trials:
    print("[%d]loss:" % l, i['result']['loss'])
    print(i['misc']['vals'])
    l = l + 1
