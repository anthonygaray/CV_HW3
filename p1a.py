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
from c import LFW, Net
import sys

# Hyper Parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# LFW Dataset
train_dataset = LFW('train.txt', transform=transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))
test_dataset = LFW('test.txt', transform=transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


if (len(sys.argv) < 2):
    print ('Error: Please enter an argument')

elif(sys.argv[1] == '--save'):

    if (len(sys.argv) == 3):

        net = Net()
        net.cuda()

        # Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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
                merged_out = torch.cat([output1, output2], 1)

            last_layer = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
            last_layer.cuda()
            out = last_layer(merged_out)

            optimizer.zero_grad()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch {}\n Current loss {}\n".format(epoch, loss.data[0]))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.data[0])

        # show_plot(counter,loss_history)

        # Save the Trained Model
        torch.save(net.state_dict(), sys.argv[2])

    else:
        print ("Error: File name needed")

elif (sys.argv[1] == '--load'):

    if (len(sys.argv) == 3):

        net = Net()
        net.load_state_dict(torch.load(sys.argv[2]))

        # Test the Model
        # net.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
        # correct = 0
        # total = 0

        # for i, data in enumerate(test_loader):
        #	img1, img2, label = data
        #	img1 = Variable(img1).cuda()
        #	img2 = Variable(img2).cuda()
        #	out1, out2 = net(img0, img1)

        #	_, predicted = torch.max(outputs.data, 1)

        # for images, labels in test_loader:
        #    images = Variable(images).cuda()
        #    outputs = cnn(images)
        #    _, predicted = torch.max(outputs.data, 1)
        #    total += labels.size(0)
        #    correct += (predicted.cpu() == labels).sum()

        # print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    else:
        print ("Error: File name needed")