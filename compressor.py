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

class Compressor(nn.Module):
    def __init__(self, size_of_image, compressedSize):
        super(Compressor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.dropOut = nn.Dropout()
        self.lin1 = nn.Linear(32 * (size_of_image // 4) * (size_of_image // 4), 1000)
        self.lin2 = nn.Linear(1000, compressedSize)

    def forward(self, x):
        x, index1 = self.layer1(x)
        x, index2 = self.layer2(x)
        x = self.dropOut(x)
        x = F.leaky_relu(self.lin1(x.view(x.shape[0], 1, -1)))
        x = self.lin2(x)
        return (x, index1, index2)

class Decompressor(nn.Module):
    def __init__(self, size_of_final_image, compressedSize):
        self.size_of_final = size_of_final_image
        super(Decompressor, self).__init__()
        self.lin1 = nn.Linear(compressedSize, 1000)
        self.lin2 = nn.Linear(1000, 32 * (size_of_final_image // 4) * (size_of_final_image // 4))
        self.dropOut = nn.Dropout(p=0.2)
        self.unpool3 = nn.MaxUnpool2d(2, padding=2)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5),
            nn.LeakyReLU())
        self.unpool4 = nn.MaxUnpool2d(2, padding=2)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=5),
            nn.LeakyReLU())

    def forward(self, x, index1, index2):
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.dropOut(x)
        x = self.unpool3(x.view(x.shape[0], 32, self.size_of_final // 4, self.size_of_final // 4), index2)
        x = self.layer3(x)
        x = self.unpool4(x, index1)
        x = self.layer4(x)
        return x

compressor = Compressor(28, 200).to(device)
compressor.load_state_dict(torch.load("compressor.state"))
decompressor = Decompressor(28, 200).to(device)
decompressor.load_state_dict(torch.load("decompressor.state"))
learningRate = 0.01

index = random.randint(0, len(mnistDataTest))

compressor.eval()
decompressor.eval()

inPic = mnistDataTest.train_data[index].float().view(1, 1, 28, 28)

with torch.no_grad():
    compressed, index1, index2 = compressor(inPic.to(device))
    print(compressed.shape)
    print(index1.shape)
    print(index2.shape)
    decompressed = decompressor(compressed, index1, index2)
    cv2.imshow("original", inPic.squeeze(0).squeeze(0).to("cpu").numpy())
    cv2.imshow("new", decompressed.squeeze(0).squeeze(0).to("cpu").numpy())


compOptimizer = optim.SGD(compressor.parameters(), lr=learningRate)
decompOptimizer = optim.SGD(decompressor.parameters(), lr=learningRate)

compScheduler = optim.lr_scheduler.ReduceLROnPlateau(compOptimizer, patience = 2)
decompScheduler = optim.lr_scheduler.ReduceLROnPlateau(decompOptimizer, patience = 2)
blank = np.zeros((400, 400, 3), np.uint8)
lossArray = []

trainLoader = data.DataLoader(mnistDataTrain, batch_size=1028, shuffle=True)
testLoader = data.DataLoader(mnistDataTest, batch_size=1028, shuffle=False)

torch.autograd.set_detect_anomaly(True)

done = False

while True:
    for i, (inputs, _) in enumerate(trainLoader, 0):
        inputs = Variable(inputs.to(device))
        checkVal = inputs.clone().detach()
        checkVal.requires_grad = False
        compressor.train(True)
        decompressor.train(True)
        compOptimizer.zero_grad()
        decompOptimizer.zero_grad()
        inputs = inputs.float().view(-1, 1, 28, 28)
        compressed, index1, index2 = compressor(inputs)
        decompressed = decompressor(compressed, index1, index2)
        lossFn = nn.MSELoss()
        loss = lossFn(decompressed, inputs)
        print("Training loss: " + str(loss.item()))
        loss.backward()
        compOptimizer.step()
        decompOptimizer.step()
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
            compressor.train(False)
            decompressor.train(False)
            compressed, index1, index2 = compressor(imagesVal)
            decompressed = decompressor(compressed, index1, index2)
            lossFn = nn.MSELoss()
            lossSum += lossFn(decompressed, imagesVal).item()
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

compressor.eval()
decompressor.eval()

inPic = mnistDataTest.train_data[index].float().view(1, 1, 28, 28)

with torch.no_grad():
    compressed, index1, index2 = compressor(inPic.to(device))
    print(compressed.shape)
    print(index1.shape)
    print(index2.shape)
    decompressed = decompressor(compressed, index1, index2)
    cv2.imshow("original", inPic.squeeze(0).squeeze(0).to("cpu").numpy())
    cv2.imshow("new", decompressed.squeeze(0).squeeze(0).to("cpu").numpy())

def doSave():
    torch.save(compressor.state_dict(), "compressor.state")
    torch.save(decompressor.state_dict(), "decompressor.state")
