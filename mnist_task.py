import torch
import random
import numpy as np

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

import torchvision.datasets

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()

class MNISTNet(torch.nn.Module):

    def conv_act(self, in_channels, out_channels, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.001)
        )

    def full_convolution_block(self, ch_gradation, padding=(0, 0)):
        return torch.nn.Sequential(
            self.conv_act(ch_gradation[0], ch_gradation[1], padding=padding[0]),
            self.conv_act(ch_gradation[1], ch_gradation[2], padding=padding[1]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __init__(self):
        super(MNISTNet, self).__init__()

        channel_gradation = [1, 10, 20, 24, 16]

        self.full_convolution_block1 = self.full_convolution_block(channel_gradation[0:])
        self.full_convolution_block2 = self.full_convolution_block(channel_gradation[2:], padding=(1, 0))

        self.full_connected_block = torch.nn.Sequential(
            torch.nn.Linear(5 * 5 * 16, 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.full_convolution_block1(x)
        x = self.full_convolution_block2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.full_connected_block(x)
        return x

mnist_net = MNISTNet()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_net = mnist_net.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=3.0e-4)

batch_size = 100

X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(300):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds = mnist_net(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(X_test)
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu().item()
    print(accuracy)
