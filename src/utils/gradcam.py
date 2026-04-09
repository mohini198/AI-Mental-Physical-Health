import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple → take first element
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_image, class_idx):

        # Ensure clean tensor
        input_image = input_image.clone().detach()

        # Forward pass
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Target class score
        loss = output[:, class_idx]

        # Backward pass
        loss.backward()

        # Safety checks
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured properly")

        # Convert to numpy safely
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Compute weights (global average pooling)
        weights = np.mean(gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize safely
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # Resize to match input
        cam = cv2.resize(cam, (224, 224))

        return cam
