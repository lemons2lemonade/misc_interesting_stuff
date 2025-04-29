import torch
import coremltools as ct
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
from scipy.spatial.distance import cosine, euclidean
import numpy as np
import os
import re
import warnings
from torchvision import transforms
import transformers
from transformers import ViTModel
import torch.nn.utils.prune as prune
import psutil

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024 * 1024):.2f} MB")  # RSS is the Resident Set Size

# Vision Transformer (ViT) Model without classification head
class SimpleViT(nn.Module):
    def __init__(self):
        super(SimpleViT, self).__init__()
        self.vit = AutoModelForImageClassification.from_pretrained("google/vit-large-patch16-384")
        self.vit.head = nn.Identity()  # remove classification head

    def forward(self, x):
        return self.vit(x)  # return embeddings from ViT

# Specify transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the model
model = SimpleViT()
# Load the saved model weights
checkpoint = torch.load("/Volumes/datasets/pokemon-tcg-data/first_checkpoint.pth", map_location=torch.device('cpu'))
print("LOADED MODEL...")
model.load_state_dict(checkpoint['model_state_dict'])
print("LOADED STATE DICT...")
# Set the model to evaluation mode
model.eval()
print_memory_usage()

module = model.vit.vit.encoder.layer[0].attention.attention.query
prune.random_unstructured(module, name='weight', amount=0.6)
print("ITEMS SELECTED FOR PRUNING...")
# Make pruning permanent for the pruned module
prune.remove(module, 'weight')
print("PRUNING COMPLETE...")
print_memory_usage()

# Create a dummy input tensor matching the input size of your model
dummy_input = torch.randn(1, 3, 384, 384)  # Adjust the size according to your ViT model's requirements

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "/Volumes/datasets/pokemon-tcg-data/model.onnx", opset_version=11,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("SAVED ONNX MODEL...")
# Load the ONNX model
onnx_model = ct.onnx.load_model("/Volumes/datasets/pokemon-tcg-data/model.onnx")
print("LOADED ONNX MODEL...")
# Convert the model to Core ML
coreml_model = ct.convert(onnx_model, minimum_ios_deployment_target='13')
print("CONVERTED ONNX MODEL TO MLX MODEL...")
# Save the model in MLX format
coreml_model.save("/Volumes/datasets/pokemon-tcg-data/model.mlmodel")