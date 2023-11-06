from d2l import d2l
import torch
from torch import nn

features, labels = d2l.synthetic_data(torch.tensor([2, -3.4]), 4.2, 1000)
batch_size = 10
data_iter = d2l.load_array((features, labels), batch_size)
net = nn.Sequential(nn.Linear(2, 1))
# print(len(net))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss = nn.MSELoss()
loss = nn.HuberLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    l = loss(net(features), labels)
    print(f"epoch {epoch+1},loss{l:f}")