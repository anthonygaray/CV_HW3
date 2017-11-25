import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from PIL import Image
import torch.nn as nn
import torch.utils.data as data

class LFW(data.Dataset):

    def __init__(self, data_file, transform=None):

        self.data_file = data_file
        self.transform = transform

        values = open(self.data_file).read().split()

        self.data_loc = [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]
	self.img_data = ''

    def __getitem__(self, index):

        img1 = Image.open('lfw/' + self.data_loc[index][0])
        img2 = Image.open('lfw/' + self.data_loc[index][1])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
	
	int_label = int(self.data_loc[index][2])
	arr_label = np.array([int_label])
        self.img_data = (img1, img2, torch.from_numpy(arr_label))

        return self.img_data

    def __len__(self):
        return len(self.data_loc)


# Hyper Parameters
num_epochs = 5
batch_size = 100

# LFW Dataset
train_dataset = LFW('train.txt', transform=transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()]))

test_dataset = LFW('test.txt', transform=transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # set up convolution layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )

    def forward_once(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

net = Net()
net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

counter = []
loss_history = []
iteration_number = 0

# Train the Model
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        img0, img1, label = data
	img0 = Variable(img0).cuda()
	img1 = Variable(img1).cuda()
	label = Variable(label).cuda()

        # Forward + Backward + Optimize
	output1, output2 = net(img0, img1)
        optimizer.zero_grad()

        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

	if i % 10 == 0:
		print("Epoch {}\n Current loss {}\n".format(epoch, loss.data[0]))
		iteration_number += 10

# Save the Trained Model
torch.save(net.state_dict(), 'net.pkl')
