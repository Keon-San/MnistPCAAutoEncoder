# MnistPCAAutoEncoder
In this project, I trained an Autoencoder on the MNIST dataset. Then, I performed PCA on its representation of the images. This project was inspired by [this video](https://www.youtube.com/watch?v=NTlXEJjfsQU) by CaryKH, a well-known youtuber who does projects with code, especially machine learning.

To run the code, run doPca.py. You need to have Pytorch, Pandas, Numpy, Scikit Learn, and Tkinter installed, along with Python 3. It will encode all of the images, and when done, will open a GUI where you can put in the number of an image, and then edit the first 10 values corresponding to that image.
