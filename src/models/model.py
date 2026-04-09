import torch
import torch.nn as nn
from torchvision import models

def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)

    return model
