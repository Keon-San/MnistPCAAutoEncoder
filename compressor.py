import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data
from torch.autograd import Variable
import math
import random

mnistDataTrain = tv.datasets.MNIST("MNISTData/", download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnistDataTest = tv.datasets.MNIST("MNISTData/", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

device = "cuda" if torch.cuda.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self, size_of_image, compressedSize):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.decode1 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.decode2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        compressed = self.pool(x)
        x = F.relu(self.decode1(compressed))
        x = F.sigmoid(self.decode2(x))
        return x, compressed

net = Network(28, 200).to(device)
net.load_state_dict(torch.load("complicatedCompressor.state"))
learningRate = 0.001

index = random.randint(0, len(mnistDataTest))

net.eval()

inPic = mnistDataTest.train_data[index].float().view(1, 1, 28, 28)

with torch.no_grad():
    x, compressed = net(inPic.to(device))
    cv2.imshow("original", inPic.squeeze(0).squeeze(0).to("cpu").numpy())
    cv2.imshow("new", x.squeeze(0).squeeze(0).to("cpu").numpy())


compOptimizer = optim.Adam(net.parameters(), lr=learningRate)
compOptimizer.load_state_dict(torch.load("complicatedCompressor.optim.state"))

#compScheduler = optim.lr_scheduler.ReduceLROnPlateau(compOptimizer, patience = 2)
#decompScheduler = optim.lr_scheduler.ReduceLROnPlateau(decompOptimizer, patience = 2)
blank = np.zeros((400, 400, 3), np.uint8)
lossArray = []

trainLoader = data.DataLoader(mnistDataTrain, batch_size=1028, shuffle=True)
testLoader = data.DataLoader(mnistDataTest, batch_size=1028, shuffle=False)

torch.autograd.set_detect_anomaly(True)

done = False

while True:
    for i, (inputs, _) in enumerate(trainLoader, 0):
        inputs = Variable(inputs.to(device))
        net.train(True)
        compOptimizer.zero_grad()
        inputs = inputs.float().view(-1, 1, 28, 28)
        x, compressed = net(inputs)
        lossFn = nn.MSELoss()
        loss = lossFn(x, inputs)
        print("Training loss: " + str(loss.item()))
        loss.backward()
        compOptimizer.step()
        inputs.to("cpu")
        if (i%10 == 0):
            cv2.imshow("test", blank)
            k = cv2.waitKey(1)
            if k == 27:
                done = True
                break
    if done:
        break
    print("Done with epoch 1")
    with torch.no_grad():
        lossSum = 0
        numBatches = 0
        for i, (inputs, _) in enumerate(testLoader, 0):
            inputs = Variable(inputs.to(device))
            imagesVal = inputs.float().view(-1, 1, 28, 28)
            net.train(False)
            x, compressed = net(imagesVal)
            lossFn = nn.MSELoss()
            lossSum += lossFn(x, imagesVal).item()
            numBatches += 1
        
        loss = lossSum / numBatches
        print("Val Loss: "  + str(loss))
        lossArray.append(loss)
        #compScheduler.step(loss)
        #decompScheduler.step(loss)

    cv2.imshow("test", blank)
    k = cv2.waitKey(1)
    if k == 27:
        break

plt.plot(lossArray)
plt.show()

index = random.randint(0, len(mnistDataTest))

net.eval()

inPic = mnistDataTest.train_data[index].float().view(1, 1, 28, 28)

with torch.no_grad():
    x, compressed = net(inPic.to(device))
    cv2.imshow("original", inPic.squeeze(0).squeeze(0).to("cpu").numpy())
    cv2.imshow("new", x.squeeze(0).squeeze(0).to("cpu").numpy())

def doSave():
    torch.save(net.state_dict(), "complicatedCompressor.state")
    torch.save(compOptimizer.state_dict(), "complicatedCompressor.optim.state")
