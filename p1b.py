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
from c import LFW, Net, ContrastiveLoss
import sys
from random import *

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.01

def get_prob(prob):
    num = randint(1, 10)

    if (num >= 1 and num <= prob):
        return True;
    elif (num > prob and num <= 10):
        return False


def apply_transforms(img):

    # CONVERT TO PIL
    I = 255 * img.numpy()
    I = I.astype('uint8')
    conv_to_PIL = transforms.ToPILImage()
    im = conv_to_PIL(I)

    # ROTATE
    if (get_prob(5)):
        angle = randint(-30, 30)
        im = im.rotate(angle)

    # FLIP
    if (get_prob(5)):
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # TRANSLATE
    if (get_prob(5)):
        horiz = randint(-10, 10)
        vert = randint(-10, 10)
        im = im.transform(im.size, Image.AFFINE, (1, 0, horiz, 0, 1, vert))  # c is l/r and f is u/d

    # SCALE
    if (get_prob(5)):
        scale = uniform(0.7, 1.3)
        new_dim = int(128 * scale)

        im_resize = im.resize((new_dim, new_dim))
        im = Image.new("RGB", (128, 128), "black")
        h, w = im_resize.size
        im.paste(im_resize, (0, 0))

    # BACK TO TENSOR
    conv_to_tensor = transforms.ToTensor()
    tr_img = conv_to_tensor(im)

    return tr_img

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

    if (len(sys.argv) >= 3):

        net = Net(bin=False)
        net.cuda()

        # Loss and Optimizer
        criterion = ContrastiveLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        counter = []
        loss_history = []
        iteration_number = 0

        # Train the Model
        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader):
                img1, img2, label = data

                if (len(sys.argv) == 4):

                    if (sys.argv[3] == '--aug'):

                        for ind, img in enumerate(img1):

                            if (get_prob(7)): # If prob then apply transforms

                                # Permute
                                img = img.permute(1, 2, 0)

                                # Apply random transforms
                                tr_img = apply_transforms(img)

                                # Add tensor to img array
                                img1[ind] = tr_img

                        for ind, img in enumerate(img2):

                            if (get_prob(7)):  # If prob then apply transforms

                                # Permute
                                img = img.permute(1, 2, 0)

                                # Apply random transforms
                                tr_img = apply_transforms(img)

                                # Add tensor to img array
                                img2[ind] = tr_img

                img1 = Variable(img1).cuda()
                img2 = Variable(img2).cuda()
                label = Variable(label).cuda()

                # Forward + Backward + Optimize
                out1, out2 = net(img1, img2)

                optimizer.zero_grad()
                contrastive_loss = criterion(out1, out2, label)
                contrastive_loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    print("Epoch {}\n Current loss {}\n".format(epoch, contrastive_loss.data[0]))
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(contrastive_loss.data[0])

        # show_plot(counter,loss_history)

        # Save the Trained Model
        torch.save(net.state_dict(), sys.argv[2])

    else:
        print ("Error: File name needed")

elif (sys.argv[1] == '--load'):

    if (len(sys.argv) >= 3):

        net = Net(bin=False)
        net.cuda()
        net.load_state_dict(torch.load(sys.argv[2]))
        correct = 0
        total = 0
        thresh = 0.5

        #TODO enter testing code for dissimilarity

        for i, data in enumerate(train_loader):
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
