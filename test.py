import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


model = TestNet()
model.eval()
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.float)
out = model(data)
print(out)
loss_fn = nn.MSELoss()
label = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
temp = 0
for i in range(len(label)):
    temp += (out[i] - label[i]) ** 2
loss = loss_fn(out, label)
print(loss)
print(temp / 5)
