# Torch2TF_Weight_Convert
convert torch model weight to tensorflow one, try without adding transpose layers


## installation

Install proper version of `torch`, `torchvision`, that fit the environment, cuda version, etc. 

See torch official [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

Install `tensorflow2`

Install `matplotlib`

Switch `numpy` version if needed.

## Exmaple

Following example model are from [torch official CNN tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

Example torch model like
```python
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
```

Corresponding tensorflow model(prepare ourselves)
```python
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
```

Assume torch model trained well, prepared, variable named as `net`, eg:
```python
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
```


Tensorflow model variable named as `net_tf`, eg:
```python
net_tf = Net_TF()
```

layer `fc1` is the first linear layer in model process, so need provide input shape for layer `fc1`

Explore layer input/output shapes from torchsummary.

Then arrange as following:
```python
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
```

Finally call our convert function

```python
from torch_tf_convert import convert

convert(torch_list, tf_list, shape_list)
```

Tensorflow model variable `net_tf` are ready to do the same things as torch one


### simple example of running convert function

```bash
python example.py
```

Demonstrate that torch output will be very close to tensorflow one.

Or run on Colab, to further explore:

[Torch_Convert_TF](https://github.com/KuoYuChang/Colab_MIT/blob/main/tools/Torch_Convert_TF.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_MIT/blob/main/tools/Torch_Convert_TF.ipynb)