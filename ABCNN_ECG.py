import torch
import torch.nn as nn
import torch.utils.data as Data
import pickle
import numpy as np
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, classification_report
from torch.nn.parameter import Parameter


def one_hot(y_, maxvalue = None):
    if maxvalue==None:
        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
    else:
        n_values = maxvalue
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

EPOCH = 1000
LR = 0.001

# load datasets
data = pickle.load(open( '../ECG_code/MIT_BIH_data_5class.p', "rb" ), encoding='latin1')
n_class=5
# print(data.shape)

data = data[:, :-1]
n_fea = data.shape[-1] - 1

# normalization
feature_all = data[:, 0:n_fea]
label_all = data[:, n_fea:n_fea+1]
feature_normalized=preprocessing.scale(feature_all)
all = np.hstack((feature_normalized, label_all))

n_seg = all.shape[0]
np.random.shuffle(all)
all = torch.Tensor(all).to(device)

# split train and test data,train
ratio = 0.8
train_data = all[: int(ratio*n_seg)]
test_data = all[int(ratio*n_seg):]

BATCH_SIZE = int(all.shape[0]*0.05)


train_x_3d = train_data[:, 0:480].reshape([-1, 2, 240])
train_x_3d = torch.unsqueeze(train_x_3d, dim=1)

train_y_ = train_data[:, 480:481].long().to(device)


train_data = Data.TensorDataset(train_x_3d, train_y_)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

test_data_3d = test_data[:, 0:480].reshape([-1, 2, 240])
test_data_3d = torch.unsqueeze(test_data_3d, dim=1)

test_y = test_data[:, 480:481].long().to(device)


class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=[2, 2],
                stride=1,
                padding=[1,2],
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[1, 2])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, [2, 2], 1, padding=[1,2]),
            nn.ReLU(),
            nn.MaxPool2d([1,2])
        )

        self.fc = nn.Linear(1984, 200)
        self.out = nn.Linear(200, 5)
        self.weight = Parameter(torch.Tensor(2, 32))


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc(x))
        x = F.dropout(x, 0.2)

        output = self.out(x)
        return output, x

cnn = CNN()
cnn.to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
import time


best_acc = 0
best_auc = 0
time_s = time.perf_counter()
# training & testing
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = cnn(train_x)[0]

        loss = loss_func(output, train_y.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step == 0:
            test_output, last_layer = cnn(test_data_3d)
            test_loss = loss_func(test_output, test_y.squeeze(1))

            test_y_score = one_hot(test_y.data.cpu().numpy())
            pred_score = F.softmax(test_output, dim=0).data.cpu().numpy()
            auc_score = roc_auc_score(test_y_score, pred_score)


            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            pred_train = torch.max(output, 1)[1].data.cpu().numpy()
            train_acc = float((pred_train == train_y.squeeze(1).data.cpu().numpy()).astype(int).sum()) / float(train_y.size(0))
            test_acc = float((pred_y == test_y.squeeze(1).data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| STEP: ', step, '|train loss: %.4f' % loss.item(),
                  ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(),
                  'test ACC: %.4f' % test_acc,
                  '| AUC: %.4f' % auc_score)


            """Save model"""
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(cnn.state_dict(), '../ECG_code/saved_model/ECG_ours.pt')
                print('model saved, for auc: {}'.format(best_auc))

"""load best model"""
cnn.load_state_dict(torch.load('../ECG_code/saved_model/ECG_ours.pt'))
cnn.eval()
test_output = cnn(test_data_3d)[0]

test_y_score = one_hot(test_y.data.cpu().numpy())
pred_score = F.softmax(test_output, dim=0).data.cpu().numpy()
pred_y = torch.argmax(torch.from_numpy(pred_score), dim=1).data.cpu().numpy()

auc_score = roc_auc_score(test_y_score, pred_score)

print('===========================================================')

print(classification_report(test_y.data.cpu().numpy(), pred_y, digits=4))
print("AUC: ", auc_score)







