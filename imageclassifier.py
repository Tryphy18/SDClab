import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# 🔨 Load Pre-trained ResNet Model
model = torchvision.models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# ✅ Define the Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 🔍 Download and preprocess a test image
image_path = "path_to_your_image.jpg"  # Replace with your image path
img = Image.open(image_path)
img = transform(img).unsqueeze(0)  # Add batch dimension

# 🧠 Predict using the model
with torch.no_grad():
    output = model(img)
    _, predicted_class = torch.max(output, 1)

# 🏷️ Load ImageNet class labels (1000 classes)
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
import requests
import json

response = requests.get(LABELS_URL)
class_idx = json.loads(response.text)

# Decode the class
predicted_label = class_idx[str(predicted_class.item())][1]

# Show the image and prediction
plt.imshow(Image.open(image_path))
plt.title(f"Prediction: {predicted_label}")
plt.axis('off')
plt.show()

print(f"Predicted Label: {predicted_label}")
