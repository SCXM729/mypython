import d2l
import torchvision
import torch
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True
)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fasion_mnnnist_labels(y))
batch_size = 256
train_iter = data.DataLoader(
    mnist_train, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers()
)
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f"{timer.stop()} sec")


num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(x):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
