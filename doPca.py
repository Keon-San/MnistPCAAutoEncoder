import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision as tv
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tkinter import *
import cv2

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

        return y

    def deCompress(self, x):
        x = F.leaky_relu(self.layer4(x))
        x = F.leaky_relu(self.layer5(x))
        x = F.sigmoid(self.layer6(x))
        return x

net = Network(28, 50).to(device).float()
net.load_state_dict(torch.load("simpleNet.state"))

dictData = {}

for i in range(0, len(mnistDataTrain)):
    compressed = net(mnistDataTrain[i][0].to(device).float().view(-1, 1, net.lenImage))
    dictData[i] = compressed.to("cpu").detach().numpy()
    if i%100 == 0:
        print(i)
for i in range(len(mnistDataTrain), len(mnistDataTrain)+len(mnistDataTest)):
    compressed = net(mnistDataTest[i-len(mnistDataTrain)][0].to(device).float().view(-1, 1, net.lenImage))
    dictData[i] = compressed.to("cpu").detach().numpy()
    if i%100 == 0:
        print(i)

data = np.array(list(dictData.items()))

scalar = StandardScaler()

data = data.tolist()

for x in range(0, len(data)):
    data[x] = data[x][1][0][0].tolist()

dataScaled = scalar.fit_transform(data)

pca = PCA(n_components=50)
principalComponents = pca.fit_transform(dataScaled)

window = Tk()
idEntry = Entry(window)
idEntry.pack()

currPCASelection = None

scales = []

def changeId():
    global currPCASelection
    currPCASelection = principalComponents[int(idEntry.get())].copy()
    for x in range(0, 10):
        scales[x].set(currPCASelection[x])
    drawPCA()

def doPCA():
    global currPCASelection
    for x in range(0, 10):
        currPCASelection[x] = float(scales[x].get())
    drawPCA()

def drawPCA():
    normalImg = net.deCompress(torch.tensor(scalar.inverse_transform(pca.inverse_transform(currPCASelection))).to(device).float()).squeeze(0).view(28,28).to("cpu").detach().numpy()
    cv2.imshow("image", normalImg)
    

idSubmit = Button(window, text="Goto", command=changeId)
idSubmit.pack()
for x in range(0, 10):
    entry = Entry(window)
    entry.pack()
    scales.append(Scale(window, from_=-.1, to_=20, resolution=0.01, orient=HORIZONTAL, tickinterval=0.5, length=1200))
    scales[x].pack()

updatePCA = Button(window, text="pca submit", command=doPCA)
updatePCA.pack()

mainloop()
