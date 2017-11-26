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
from random import *
import skimage

def get_prob():

    num = randint(1, 10)

    if (num >= 1 and num <= 7):
        return True;
    elif (num > 7 and num <= 10):
        return False


# Hyper Parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# LFW Dataset
train_dataset = LFW('train.txt', transform=transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))
test_dataset = LFW('test.txt', transform=transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

if (len(sys.argv) < 2):
    print ('Error: Please enter an argument')

elif (sys.argv[1] == '--save'):

    if (len(sys.argv) == 3):

        net = Net(bin=True)
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
                img1, img2, label = data

                # if (sys.argv[3] == '--aug'):
                #
                #     if (get_prob()):
                #

                img1 = skimage.transform.rotate(img1, angle=10, resize=False)

                img1 = Variable(img1).cuda()
                img2 = Variable(img2).cuda()
                label = Variable(label).cuda()

                # Forward + Backward + Optimize
                out = net(img1, img2)

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

    if (len(sys.argv) >= 3):

        net = Net(bin=True)
        net.cuda()
        net.load_state_dict(torch.load(sys.argv[2]))
        correct = 0
        total = 0
        thresh = 0.5

        for i, data in enumerate(test_loader):
            img1, img2, label = data
            img1 = Variable(img1).cuda()
            img2 = Variable(img2).cuda()
            out = net(img1, img2)
            predicted = out.data.cpu().numpy()
            actual = label.numpy()

            for i, val in enumerate(predicted):

                if (val[0] >= thresh):
                    round_val = 1
                else:
                    round_val = 0

                if (round_val == actual[i]):
                    correct += 1

            total += label.size(0)

        print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        print('Correct: %d' % correct)
        print('Total: %d' % total)

    else:
        print ("Error: File name needed")



