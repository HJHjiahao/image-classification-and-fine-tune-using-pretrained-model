# image-classification-and-fine-tune-using-pretrained-model

In some common Computer Vision(CV) tasks, such as object detection and image classification, deep learning-based methods have achieved good results. However, it costs much time and computational resource to train for great performance from scratch.   
In fact, a good news is that a lot of classic models have been trained effectively based on some large-size dataset, like VGG, Resnet, etc. In torchvision, the vision library of PyTorch, these models are provided for us to call directly and conveniently.

## Application without modification.
```
from torchvision import models
res50 = models.resnet50(pretrained=True)
# print(res50)  check whether the net architecture could be use directly.
```
