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

def doClean(image):
    return image

mnistDataTrain = tv.datasets.MNIST("MNISTData/", download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnistDataTest = tv.datasets.MNIST("MNISTData/", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

device = "cuda" if torch.cuda.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self, size_of_image, compressedSize):
        self.lenImage = size_of_image * size_of_image
        super(Network, self).__init__()
        self.layer1 = nn.Linear(size_of_image*size_of_image, int((size_of_image*size_of_image) - .333*(size_of_image*size_of_image)-compressedSize))
        self.layer2 = nn.Linear(int((size_of_image*size_of_image) - .333*(size_of_image*size_of_image)-compressedSize), int((size_of_image*size_of_image) - .666*(size_of_image*size_of_image)-compressedSize))
        self.layer3 = nn.Linear(int((size_of_image*size_of_image) - .666*(size_of_image*size_of_image)-compressedSize), compressedSize)
        self.layer4 = nn.Linear(compressedSize, int((size_of_image*size_of_image) - .666*(size_of_image*size_of_image)-compressedSize))
        self.layer5 = nn.Linear(int((size_of_image*size_of_image) - .666*(size_of_image*size_of_image)-compressedSize), int((size_of_image*size_of_image) - .333*(size_of_image*size_of_image)-compressedSize))
        self.layer6 = nn.Linear(int((size_of_image*size_of_image) - .333*(size_of_image*size_of_image)-compressedSize), size_of_image*size_of_image)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        y = F.leaky_relu(self.layer3(x))
        x = F.leaky_relu(self.layer4(y))
        x = F.leaky_relu(self.layer5(x))
        x = F.leaky_relu(self.layer6(x))

        return (x,y)

def doSave():
    torch.save(net.state_dict(), "simpleNet.state")
    torch.save(netOptim.state_dict(), "simpleNet.optim.state")

net = Network(28, 50).to(device)
net.load_state_dict(torch.load("simpleNet.state"))
learningRate = 0.001

index = random.randint(0, len(mnistDataTest))
inPic = mnistDataTest.train_data[index].float().view(1, 1, -1)
with torch.no_grad():
    outPut, compressed = net(inPic.to(device))
    outPut = doClean(outPut.squeeze(0).view(28,28).to("cpu").numpy())
    cv2.imshow("original", inPic.squeeze(0).view(28,28).to("cpu").numpy())
    cv2.imshow("new", outPut)

netOptim = optim.Adam(net.parameters(), lr=learningRate)
netOptim.load_state_dict(torch.load("simpleNet.optim.state"))

netScheduler = optim.lr_scheduler.ReduceLROnPlateau(netOptim, patience = 5)

blank = np.zeros((400, 400, 3), np.uint8)

lossArray = []

trainLoader = data.DataLoader(mnistDataTrain, batch_size=2056, shuffle=True)
testLoader = data.DataLoader(mnistDataTest, batch_size=2056, shuffle=False)

torch.autograd.set_detect_anomaly(True)

done = False

while True:
    for i, (inputs, _) in enumerate(trainLoader, 0):
        inputs = Variable(inputs.to(device))
        net.train(True)
        netOptim.zero_grad()
        inputs = inputs.float().view(-1, 1, net.lenImage)
        outputs, compressed = net(inputs)
        lossFn = nn.MSELoss()
        loss = lossFn(doClean(outputs), inputs)
        print("Training loss: " + str(loss.item()))
        loss.backward()
        netOptim.step()
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
            inputs = inputs.float().view(-1, 1, net.lenImage)
            net.train(False)
            outputs, compressed = net(inputs)
            lossFn = nn.MSELoss()
            lossSum += lossFn(doClean(outputs), inputs).item()
            numBatches += 1
        
        loss = lossSum / numBatches
        print("Val Loss: "  + str(loss))
        lossArray.append(loss)
        netScheduler.step(loss)

    cv2.imshow("test", blank)
    k = cv2.waitKey(1)
    if k == 27:
        break


plt.plot(lossArray)
plt.show()

index = random.randint(0, len(mnistDataTest))
inPic = mnistDataTest.train_data[index].float().view(1, 1, -1)
with torch.no_grad():
    outPut, compressed = net(inPic.to(device))
    outPut = doClean(outPut.squeeze(0).view(28,28).to("cpu").numpy())
    cv2.imshow("original", inPic.squeeze(0).view(28,28).to("cpu").numpy())
    cv2.imshow("new", outPut)


