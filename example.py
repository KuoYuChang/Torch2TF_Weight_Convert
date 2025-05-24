import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

from torch_tf_convert import convert

# from torch official guide
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# torch model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# corresponding tensorflow model
class Net_TF(tf.keras.Model):
    def __init__(self):
        super(Net_TF, self).__init__()
        self.conv1 = layers.Conv2D(6, (5, 5), activation='relu', input_shape=(None, None, 3))
        self.pool = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(16, (5, 5), activation='relu')
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(10)

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def convert_example():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # prepare for tf tensor
    images_np = images.cpu().detach().numpy()
    images_np, images_np.shape

    # create torch model
    net = Net()
    # load model weight if prepared
    #net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)

    summary(net, input_size=(3, 32, 32))

    images = images.to(device)
    outputs = net(images)
    print("torch raw outputs: ", outputs)


    # covert torch weights to tf ones
    net_tf = Net_TF()
    input_data = tf.random.uniform((1, 32, 32, 3))
    net_tf(input_data)
    net_tf.summary()

    torch_list = []
    tf_list = []
    shape_list = []


    # manually fill-in
    torch_list.append(net.conv1)
    torch_list.append(net.conv2)
    torch_list.append(net.fc1)
    torch_list.append(net.fc2)
    torch_list.append(net.fc3)

    tf_list.append(net_tf.conv1)
    tf_list.append(net_tf.conv2)
    tf_list.append(net_tf.fc1)
    tf_list.append(net_tf.fc2)
    tf_list.append(net_tf.fc3)

    shape_list.append(None)
    shape_list.append(None)
    # shape before first linear
    shape_list.append((16, 5, 5))
    shape_list.append(None)
    shape_list.append(None)

    assert len(torch_list) == len(tf_list), "number of torch layers in model must same as tensorflow layers in one"


    convert(torch_list, tf_list, shape_list)


    images_tf = tf.convert_to_tensor(images_np)
    images_tf = tf.transpose(images_tf, perm=[0, 2, 3, 1])

    tf_outputs = net_tf(images_tf)
    print("tensorflow output: ", tf_outputs)

if __name__ == "__main__":
    convert_example()