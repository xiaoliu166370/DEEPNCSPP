import os
import torch
from precess import load_data_PeNGaRoo
from model.model import DeepNCSPP
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings("ignore")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.manual_seed(24)
torch.cuda.manual_seed(24)

batch_size = 16
device = torch.device('cuda')


acc = 0
f1 = 0
pre = 0
recall = 0
auc_score = 0
mcc = 0
ks = 5
X, Xs, y, test_X, tests, test_y, vocab = load_data_PeNGaRoo(1000)
print(len(vocab))
# net = TransformerEncoder(21, 32, 32, 32, 32, [32], 32, 128, 8, 1, 0.5).to(device)
net = DeepNCSPP(21, 32, 512, 2, 0.5).to(device)

net.load_state_dict(torch.load('model.params'))
# test_iter = load_array((torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)),
#                        batch_size)


net.eval()

yts = test_y

t_X = torch.tensor(test_X, dtype=torch.long).to(device)
t_X_s = torch.tensor(tests, dtype=torch.float32).to(device)

fake_test = net(t_X, t_X_s.unsqueeze(1)).cpu().detach()

fakes = torch.argmax(fake_test, dim=1).numpy().tolist()
# torch.save(fakes,'predict_class.pkl')
fakes_p = fake_test[:, 1].numpy().tolist()
# torch.save(fakes_p,'predict_prob.pkl')
# print(fakes_p)
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


print('recall: ', recall_)
print('spe: ', spe_)
print('pre: ', pre_)
print('acc: ', acc_)
print('mcc: ', score_)
print('f1: ', f1_)
print('auc: ', auc_)
print('auprc: ', aupr_)
