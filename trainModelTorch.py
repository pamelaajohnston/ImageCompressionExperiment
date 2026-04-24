#Based on https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#This for the data loaders from directories.
# Source - https://stackoverflow.com/a/76975153
# Posted by Ebrahim Pichka, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-23, License - CC BY-SA 4.0


import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import common as c


import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # in channels, out channels, kernel size 3, 32, 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net_3x3_fc(nn.Module):
    def __init__(self):
        super().__init__()
        # input 32x32x3
        self.conv1 = nn.Conv2d(3, 32, 3) # in channels, out channels, kernel size 3, 32, 3
        # now 30x30x32
        self.conv2 = nn.Conv2d(32, 32, 3)
        # now 28x28x32
        self.conv3 = nn.Conv2d(32, 32, 3)
        # now 26x26x32 (flattening)
        self.fc1 = nn.Linear(26*26*32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net_4x4_for_compression(nn.Module):
    def __init__(self):
        super().__init__()
        # input 32x32x3
        self.conv1 = nn.Conv2d(3, 32, 4) # in channels, out channels, kernel size 4, 32, 3
        # now 29x29x32
        self.pool = nn.MaxPool2d(2, 2)
        # pooled to 14x14x32
        self.conv2 = nn.Conv2d(32, 32, 4)
        # now 11x11x32, pooled to 5x5x32 (flattening)
        self.fc1 = nn.Linear(5*5*32, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x




if __name__ == "__main__":
    img_w = 32
    img_h = 32
    doTheSaving = True

    batch_size = 32
    epochs = 20

    datadir = "UCIDsinglecompression32x32patchesDataSet"
    #datadir = "PetImages"

    traindir = '{}/train'.format(datadir)
    fileList = c.createFileList(traindir)
    trainSamples = len(fileList)
    testdir = '{}/test'.format(datadir)
    fileList = c.createFileList(testdir)
    testSamples = len(fileList)

    IMAGE_WIDTH=32
    IMAGE_HEIGHT=32
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # No normalizing
     ])

    train_dataset = ImageFolder(root=traindir, transform=transform)
    test_dataset = ImageFolder(root=testdir, transform=transform)

 

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    classes = train_dataset.classes
    print("Class names: ",classes)

    # Checking the data
    img, label = train_dataset[0][0], train_dataset[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")
        


    # get some random training images
    #dataiter = iter(train_loader)
    #images, labels = next(dataiter)
    #imshow(torchvision.utils.make_grid(images))
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



    # The network
    net = Net()

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print(inputs)
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # testing
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    
