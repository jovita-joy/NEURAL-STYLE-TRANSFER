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



## 📁 Project Structure







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


STYLE IMAGE:


OUTPUT

![Image](https://github.com/user-attachments/assets/99269b50-4d91-4e87-b152-2ede0f9aafa3)


