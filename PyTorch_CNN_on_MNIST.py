{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "PyTorch CNN on MNIST",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/yahyakesenek/KOU/blob/master/PyTorch_CNN_on_MNIST.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Lk02zqoTukUU",
        "colab_type": "text"
      },
      "source": [
        "# Training a CNN model on MNIST using PyTorch\n",
        "\n",
        "Sample MNIST images: \n",
        "\n",
        "![MNIST examples](https://www.researchgate.net/profile/Stefan_Elfwing/publication/266205382/figure/fig5/AS:267913563209738@1440886979379/Example-images-of-the-ten-handwritten-digits-in-the-MNIST-training-set.png)\n",
        "\n",
        "- 10 classes\n",
        "- 60 thousand training images\n",
        "- 10 thousand testing images\n",
        "- Each image is monochrome, 28-by-28 pixels. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bGU6NwlsXFSt",
        "colab_type": "code",
        "cellView": "both",
        "colab": {}
      },
      "source": [
        "#@title Import the required modules\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torchvision.datasets as datasets\n",
        "import torchvision.transforms as transforms"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wHPpr_Xb0fMa",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Define the \"device\". If GPU is available, device is set to use it, otherwise CPU will be used. \n",
        "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lCsBCXMwbpH5",
        "colab_type": "code",
        "outputId": "2dce5fa5-6b86-411e-ae2d-d0a77ac37d05",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        }
      },
      "source": [
        "#@title Download the dataset\n",
        "train_data = datasets.MNIST(root = './data', train = True,\n",
        "                        transform = transforms.ToTensor(), download = True)\n",
        "\n",
        "test_data = datasets.MNIST(root = './data', train = False,\n",
        "                       transform = transforms.ToTensor())"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "  0%|          | 0/9912422 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "9920512it [00:00, 21717680.86it/s]                            \n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "32768it [00:00, 331316.32it/s]\n",
            "0it [00:00, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz\n",
            "Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw\n",
            "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "1654784it [00:00, 5286097.49it/s]                           \n",
            "8192it [00:00, 128995.91it/s]\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw\n",
            "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz\n",
            "Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw\n",
            "Processing...\n",
            "Done!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n3cSd656CSZA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# About the ToTensor() transformation. \n",
        "\n",
        "# PyTorch networks expect a tensor as input with dimensions N*C*H*W  where\n",
        "# N: batch size\n",
        "# C: channel size\n",
        "# H: height\n",
        "# W: width\n",
        "\n",
        "# Normally an image is of size H*W*C. \n",
        "# ToTensor() transformation moves the channel dimension to the beginning as needed by PyTorch. "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rfDPBdnYgfGp",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#@title Define the data loaders\n",
        "batch_size = 100\n",
        "train_loader = torch.utils.data.DataLoader(dataset = train_data,\n",
        "                                             batch_size = batch_size,\n",
        "                                             shuffle = True)\n",
        "\n",
        "test_loader = torch.utils.data.DataLoader(dataset =  test_data ,\n",
        "                                      batch_size = batch_size, \n",
        "                                      shuffle = False)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fL-YXTvghaz_",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#@title Define a CNN network\n",
        "\n",
        "class CNN(nn.Module):\n",
        "    #This defines the structure of the NN.\n",
        "    def __init__(self):\n",
        "        super(CNN, self).__init__()\n",
        "        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)\n",
        "        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)\n",
        "        #self.conv2_drop = nn.Dropout2d()  #Dropout\n",
        "        self.fc1 = nn.Linear(320, 64)\n",
        "        self.fc2 = nn.Linear(64, 10)\n",
        "        self.pool = nn.MaxPool2d(2)\n",
        "        self.relu = nn.ReLU()\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.pool(self.relu(self.conv1(x)))        \n",
        "        x = self.pool(self.relu(self.conv2(x)))\n",
        "        x = x.view(-1, 320)\n",
        "        x = self.relu(self.fc1(x))\n",
        "        #x = F.dropout(x, training=self.training)\n",
        "        x = self.fc2(x)\n",
        "        return x\n",
        "\n",
        "\n",
        "# Create an instance \n",
        "net = CNN().to(device)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qMYUNXAP5p04",
        "colab_type": "code",
        "outputId": "304e7826-5859-46f4-976b-d221856b7032",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        }
      },
      "source": [
        "print(net)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CNN(\n",
            "  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))\n",
            "  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))\n",
            "  (fc1): Linear(in_features=320, out_features=64, bias=True)\n",
            "  (fc2): Linear(in_features=64, out_features=10, bias=True)\n",
            "  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
            "  (relu): ReLU()\n",
            ")\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ePLIwvAFj2zH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#@title Define the loss function and the optimizer \n",
        "loss_fun = nn.CrossEntropyLoss()\n",
        "optimizer = torch.optim.Adam( net.parameters(), lr=1.e-3)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "u75Xa5VckuTH",
        "colab_type": "code",
        "outputId": "f4016860-f2f7-493b-8254-cdea53be83db",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 527
        }
      },
      "source": [
        "#@title Train the model \n",
        "\n",
        "num_epochs = 5\n",
        "for epoch in range(num_epochs):\n",
        "  for i ,(images,labels) in enumerate(train_loader):\n",
        "    images = images.to(device)\n",
        "    labels = labels.to(device)\n",
        "    \n",
        "    optimizer.zero_grad()\n",
        "    output = net(images)\n",
        "    loss = loss_fun(output, labels)\n",
        "    loss.backward()\n",
        "    optimizer.step()\n",
        "    \n",
        "    if (i+1) % batch_size == 0:\n",
        "      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'\n",
        "                 %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.item()))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch [1/5], Step [100/600], Loss: 0.3368\n",
            "Epoch [1/5], Step [200/600], Loss: 0.2154\n",
            "Epoch [1/5], Step [300/600], Loss: 0.1388\n",
            "Epoch [1/5], Step [400/600], Loss: 0.1059\n",
            "Epoch [1/5], Step [500/600], Loss: 0.0526\n",
            "Epoch [1/5], Step [600/600], Loss: 0.1150\n",
            "Epoch [2/5], Step [100/600], Loss: 0.1930\n",
            "Epoch [2/5], Step [200/600], Loss: 0.2080\n",
            "Epoch [2/5], Step [300/600], Loss: 0.0404\n",
            "Epoch [2/5], Step [400/600], Loss: 0.0472\n",
            "Epoch [2/5], Step [500/600], Loss: 0.0700\n",
            "Epoch [2/5], Step [600/600], Loss: 0.0752\n",
            "Epoch [3/5], Step [100/600], Loss: 0.0439\n",
            "Epoch [3/5], Step [200/600], Loss: 0.0242\n",
            "Epoch [3/5], Step [300/600], Loss: 0.1390\n",
            "Epoch [3/5], Step [400/600], Loss: 0.0127\n",
            "Epoch [3/5], Step [500/600], Loss: 0.0406\n",
            "Epoch [3/5], Step [600/600], Loss: 0.0452\n",
            "Epoch [4/5], Step [100/600], Loss: 0.0362\n",
            "Epoch [4/5], Step [200/600], Loss: 0.0689\n",
            "Epoch [4/5], Step [300/600], Loss: 0.0508\n",
            "Epoch [4/5], Step [400/600], Loss: 0.0630\n",
            "Epoch [4/5], Step [500/600], Loss: 0.0072\n",
            "Epoch [4/5], Step [600/600], Loss: 0.0089\n",
            "Epoch [5/5], Step [100/600], Loss: 0.0233\n",
            "Epoch [5/5], Step [200/600], Loss: 0.0671\n",
            "Epoch [5/5], Step [300/600], Loss: 0.0110\n",
            "Epoch [5/5], Step [400/600], Loss: 0.0321\n",
            "Epoch [5/5], Step [500/600], Loss: 0.0169\n",
            "Epoch [5/5], Step [600/600], Loss: 0.0056\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DTPvMW5jHB9X",
        "colab_type": "code",
        "outputId": "1e61b3d7-06b0-4579-da11-ddaa59170728",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "#@title Run the trained model on the testing set\n",
        "\n",
        "correct = 0\n",
        "total = 0\n",
        "for images,labels in test_loader:\n",
        "  images = images.to(device)\n",
        "  labels = labels.to(device)\n",
        "  \n",
        "  out = net(images)\n",
        "  _, predicted_labels = torch.max(out,1)\n",
        "  correct += (predicted_labels == labels).sum()\n",
        "  total += labels.size(0)\n",
        "\n",
        "print('Percent correct: %.3f %%' %((100*correct)/(total+1)))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Percent correct: 98.000 %\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PyJAwcZoeRCU",
        "colab_type": "text"
      },
      "source": [
        "End of the notebook. "
      ]
    }
  ]
}