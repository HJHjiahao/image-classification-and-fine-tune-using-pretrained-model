# image-classification-and-fine-tune-using-pretrained-model

In some common Computer Vision(CV) tasks, such as object detection and image classification, deep learning-based methods have achieved good results. However, it costs much time and computational resource to train for great performance from scratch.   
In fact, a good news is that a lot of classic models have been trained effectively based on some large-size dataset, like VGG, ResNet, etc. In torchvision, the vision library of PyTorch, these models are provided for us to call directly and conveniently.

## Related information
PyTorch == 1.6.0  
My grayscale dataset links are below:  
[x_train](https://drive.google.com/file/d/1jXPXpEAWE57HshOx_m9A0bFAWRmyQR1d/view?usp=sharing), 
[y_train](https://drive.google.com/file/d/1QO2KPs0OTrn1Qzp5C7_s1PSWY-wc06q0/view?usp=sharing),   
[x_test](https://drive.google.com/file/d/1Zvln2lbhk6Aov3bg8-dk6aYKOtiFPkIs/view?usp=sharing), 
[y_test](https://drive.google.com/file/d/1O942yv17102st1tdtgjOTErbj_pnHIhh/view?usp=sharing)

## Application without modification
```
from torchvision import models
res50 = models.resnet50(pretrained=True)
# print(res50)  check whether the net architecture could be use directly.
```

## Application with modification
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
```
pretrained_dict = resnet50.state_dict()  # 'conv1.weight'
weights = pretrained_dict['conv1.weight']
# print(resnet50.conv1) Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
gray = torch.zeros(64, 1, 7, 7)  # from (64, 3, 7, 7)
for i, output_channel in enumerate(weights):
            # Gray = 0.299R + 0.587G + 0.114B
            gray[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            pretrained_dict['conv1.weight'] = gray
            resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            resnet50.load_state_dict(pretrained_dict)
```

### Make the middle layers frozen
Usually, we just need to train the last layer to fit our own dataset, which is really efficient. So there is no need to train the middle hidden layers.
```
for param in resnet50.parameters():
            param.requires_grad = False
# before the last layer.
```

### Fine-tune more layers
For each pretrained model, the parameters of first few layers contain the weights enforcing the **general features** of data/image, which should not be retrained usually.
Therefore, the last one layer/the last few layers contain the specific information of original dataset, which could be chosen to retrain.   
Take note that **the less the number of chosen layers, the better** for generalization of new model. If we retrain too many parameters again, why do we apply the pretrained model?
