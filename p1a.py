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
from skimage import data
import cv2

def get_prob():

    num = randint(1, 10)

    if (num >= 1 and num <= 7):
        return True;
    elif (num > 7 and num <= 10):
        return False

def apply_transforms(img):

	I = 255 * img.numpy()
        I = I.astype('uint8')
        im = Image.fromarray(I, 'RGB')
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

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
		print('train load', img1.shape)
                # if (sys.argv[3] == '--aug'):
                #
                #     if (get_prob()):
                #
		

		for ind, img in enumerate(img1):
			#print('img from batch', img.shape)
			img = img.permute(1, 2, 0)
			print('after permute', img.shape)
			I = 255 * img.numpy()
			I = I.astype('uint8')
			conv = transforms.ToPILImage()
			im = conv(I)
			
			im = im.rotate(10)
			im = im.transpose(Image.FLIP_LEFT_RIGHT)
			im = im.transform(im.size, Image.AFFINE, (1, 0, 5, 0, 1, 0)) # c is l/r and f is u/d
			im_resize = im.resize((102, 102))
			
			im = Image.new("RGB", (128, 128), "black")
			h, w = im_resize.size
			im.paste(im_resize, (64 - w/2, 64 - h/2))

			print('after transforms', im_resize.size)
			#cv2.imwrite('img.png', I)
			#im = rotate(I, 10, resize=False)
			#im = np.flip(im, 1).copy()
			#tf_shift = SimilarityTransform(translation=(10, 0))
			
			#i = 0
			#b, g, r = cv2.split(im)
			#print(b, g, r)

			#print(im.shape)
			
			conversion = transforms.ToTensor()
			tr_img = conversion(im)
			print('back to tensor', tr_img.shape)
			img1[ind] = tr_img
			
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



