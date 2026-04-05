import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # hooks
        target_layer.register_full_backward_hook(self.save_gradient)
        target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_image, class_idx):

   
        input_image = input_image.clone().detach()

        self.model.zero_grad()

        output = self.model(input_image)
        loss = output[:, class_idx]

       
        loss.backward(retain_graph=True)

        gradients = self.gradients.detach().cpu().clone().numpy()[0]
        activations = self.activations.detach().cpu().clone().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam**2
        cam = cv2.resize(cam, (224, 224))

        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam