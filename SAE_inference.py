import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from torch import nn
from SAE_train import Autoencoder, filter_none
import json
import os
import pandas as pd

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(AutoencoderKL).to(device)

ckpt = torch.load('model_epoch_25.pth')
model.load_state_dict = ckpt['model_state_dict']

#load samples
with open('../ACS_results.json', 'r') as f:
    data_info = json.load(f)

filtered_data_info = filter_none(data_info)


#load images
transform = transforms.Compose([
    transforms.ToTensor(),
])
image_dir = '../rgb'
image_files = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

latent_spaces = []
image_names = []
with torch.no_grad():
    model.eval()
    for img_file in image_files:
        if img_file not in filtered_data_info: continue
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        _, _, mu, _ = model(image)
        latent_space = mu.squeeze().cpu().numpy()

        latent_spaces.append(latent_space)
        image_names.append(img_file)

df = pd.DataFrame(latent_spaces, index=image_names)
df.to_csv('latent_spaces.csv')
