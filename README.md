# image-classification-and-fine-tune-using-pretrained-model

In some common Computer Vision(CV) tasks, such as object detection and image classification, deep learning-based methods have achieved good results. However, it costs much time and computational resource to train for great performance from scratch.   
In fact, a good news is that a lot of classic models have been trained effectively based on some large-size dataset, like VGG, Resnet, etc. In torchvision, the vision library of PyTorch, these models are provided for us to call directly and conveniently.

## Application without modification.
```
from torchvision import models
res50 = models.resnet50(pretrained=True)
# print(res50)  check whether the net architecture could be use directly.
```

## Application with modification.
### Modify the last layer
```
fc_inputs = resnet50.fc.in_features  # 2048
resnet50.fc = nn.Sequential(  # It is common to add a DNN, the layer's number depends on the user.
            nn.Linear(fc_inputs, 256),
            # nn.BatchNorm1d(256),  # To a certain extent, Batch Normalization could avoid overfitting, which is optional.
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),  # number of categories: 3 here
)
```
### Modify the first layer
To meet the requirement of our own dataset, the first layer need to be modified sometimes. For instance, we need to change the input channel if the data are grayscale images.
