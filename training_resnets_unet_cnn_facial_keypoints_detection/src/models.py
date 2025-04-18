from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torchvision import models

def _get_conv_output_size(filter_size):
    # Starting with input image dimensions
    h, w = 224, 224
    
    # Apply each conv and pooling layer
    # Conv1
    h = (h - filter_size) + 1
    w = (w - filter_size) + 1
    # Pool1
    h, w = h // 2, w // 2
    
    # Conv2
    h = (h - filter_size) + 1
    w = (w - filter_size) + 1
    # Pool2
    h, w = h // 2, w // 2
    
    # Conv3
    h = (h - filter_size) + 1
    w = (w - filter_size) + 1
    # Pool3
    h, w = h // 2, w // 2
    
    # Conv4
    h = (h - filter_size) + 1
    w = (w - filter_size) + 1
    # Pool4
    h, w = h // 2, w // 2
    
    # Conv5
    h = (h - filter_size) + 1
    w = (w - filter_size) + 1
    # Pool5
    h, w = h // 2, w // 2
    
    # Final output size
    return 512 * h * w

class SimpleNet(nn.Module):
    def __init__(self, filter_size=4):
        super(SimpleNet, self).__init__()

        # input image : 1 x 224 x 224, grayscale squared images

        self.conv1 = nn.Conv2d(1, 32, filter_size)  # 32*(filter_size, filter_size) filter ==> 221*221*32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 110*110*32
        self.dropout1 = nn.Dropout(p=0.1)

        # TODO: add more layers
        # every layer will have twice the number of filters for convolution
        # The dropout rates increase appropriately as you go deeper
        self.conv2 = nn.Conv2d(32, 64, filter_size) # 64*(filter_size, filter_size) filter ==> 107*107*64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # pool (2,2) ==> 53*53*64
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, filter_size) # 128*(filter_size, filter_size) filter ==> 50*50*128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # pool (2,2) ==> 25*25*128
        self.dropout3 = nn.Dropout(p=0.3)
        self.conv4 = nn.Conv2d(128, 256, filter_size) # 256*(filter_size, filter_size) filter ==> 22*22*256
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) # pool (2,2) ==> 11*11*256
        self.dropout4 = nn.Dropout(p=0.4) 
        self.conv5 = nn.Conv2d(256, 512, filter_size) # 512*(filter_size, filter_size) filter ==> 8*8*512
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2) # pool (2,2) ==> 4*4*512
        self.dropout5 = nn.Dropout(p=0.5)
        
        # The fully connected layers reduce dimensions gradually
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(_get_conv_output_size(filter_size), 5000) # 512*4*4 ==> 5000
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5000, 1000)
        
        # relu activations
        self.relu = nn.ReLU()
        
        self.fc3 = nn.Linear(1000, 136)

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        
        # TODO: implement forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool5(x)
        x = self.dropout5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout6(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x




class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136) # removes the classification head and replaces it with a regression head

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Resnet18Grayscale(nn.Module):
    def __init__(self):
        super(Resnet18Grayscale, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136) # removes the classification head and replaces it with a regression head

    def forward(self, x):
        # DONETODO: modify resnet18 to grayscale
        # x is a grayscale image
        # Convert to 3 channels by repeating the grayscale channel
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet18(x)
        return x

class Dinov2_grayscale(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super(Dinov2_grayscale, self).__init__()
        self.dino2 = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        n_inputs = self.dino2.embed_dim
        self.regession_head = nn.Linear(n_inputs, 136)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels
        x = self.dino2(x)
        x = self.regession_head(x)
        return x
    
if __name__ == "__main__": 
    net = Dinov2_grayscale()