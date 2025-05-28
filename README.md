# NEURAL-STYLE-TRANSFER
"Company":CODTECH IT SOLUTIONS

"NAME": JOVITA JOY

"INTERN ID":CT04DL1217

"DOMAIN":ARTIFICIAL INTELLIGENCE

"DURATION": 4 WEEKS

"MENTOR":Neela Santhosh


# 🎨 Neural Style Transfer (NST) in PyTorch

This project implements **Neural Style Transfer** using PyTorch, allowing you to apply the artistic style of one image to another. It’s based on the seminal paper *"A Neural Algorithm of Artistic Style"* by Gatys et al.



## 🧠 What is Neural Style Transfer?

Neural Style Transfer (NST) is a deep learning technique that blends two images:
- **Content Image**: The image you want to apply the style to.
- **Style Image**: The artistic image whose style you want to use.

NST generates a new image that maintains the structure of the content image but adopts the style (colors, brushstrokes, textures) of the style image.



## 📁 Project code

1.Imports and Setup

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

Fixes a common DLL issue on some Windows machines (used with matplotlib or torch).

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import transforms, models

from PIL import Image

import matplotlib.pyplot as plt

import copy

Loads PyTorch, optimization tools, image transformation tools, and pre-trained models.

2. Device and Image Size Configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128

3. Image Loader

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

Defines image transformations: resize and convert to tensor.

def image_loader(image_name):

    image = Image.open(image_name)

    image = loader(image).unsqueeze(0)
    
    return image.to(device, torch.float)

Loads and processes an image so it can be passed through a neural network.

4. Display Function

def imshow(tensor, title=None):

Converts tensor back to image and displays it with matplotlib.

5. Image Inputs

content_img = image_loader("content.jpg")

style_img = image_loader("style.jpg")

assert content_img.size() == style_img.size(), "Images must be the same size"

Loads both images and ensures they’re the same size.

6. Loss Functions
7. 
a. Content Loss

class ContentLoss(nn.Module):
    
Measures the difference between the content of the input image and the content image.

b. Style Loss

def gram_matrix(input): 

class StyleLoss(nn.Module):
    
Uses Gram Matrix to capture texture/style, then calculates style loss.

7. VGG19 Model

cnn = models.vgg19(pretrained=True).features.to(device).eval()

Loads a pre-trained VGG19 network and freezes it for feature extraction.


class Normalization(nn.Module): 

Normalizes input image using VGG's training data mean and std.

8. Building the Model with Loss Layers

def get_style_model_and_losses(...)

Creates a model that:

Extracts features from specific layers

Adds ContentLoss and StyleLoss at appropriate points

9. Style Transfer Optimization

def run_style_transfer(...)

Optimizes the input image to minimize style + content loss.

Uses the L-BFGS optimizer for better image generation.

Prints loss values every 50 steps for tracking progress.

10. Run and Display Result

input_img = content_img.clone()

output = run_style_transfer(...)

imshow(output, title="Output Image")

Clones the content image as the starting point, runs style transfer, and displays the final result.

## 🛠️ Requirements

Install dependencies using pip:

pip install torch torchvision matplotlib pillow

To use the CPU version of PyTorch (recommended if you don’t have CUDA):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

How to Run

1.Place your content.jpg and style.jpg in the project folder.

2.Run the script:

python NST.py

3.The output will be displayed using matplotlib.

4.The final image is a fusion of the two: the structure of the content with the texture of the style.

Customization

Change the style_weight and content_weight in the script to adjust the intensity of the style.

Modify the image paths to use your own photos.

References

A Neural Algorithm of Artistic Style (Gatys et al.)

PyTorch Documentation

📄 License
This project is open-source and available under the MIT License.

CONTENT IMAGE :

![Image](https://github.com/user-attachments/assets/95851ad3-0e27-4931-8342-b0c266729bb5)

STYLE IMAGE:

![Image](https://github.com/user-attachments/assets/8c16d7a5-9694-4ec7-8ca2-fb33c5412c3f)

OUTPUT

![Image](https://github.com/user-attachments/assets/99269b50-4d91-4e87-b152-2ede0f9aafa3)


