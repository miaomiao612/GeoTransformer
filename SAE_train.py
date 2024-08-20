import torch
import os
import json
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import lpips
from diffusers import AutoencoderKL



class Autoencoder(AutoencoderKL):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        self.vae = AutoencoderKL.from_pretrained("./stable-diffusion-v1-4", subfolder="vae", low_cpu_mem_usage=True)

        self.encoder = self.vae.encoder
        self.decoder = self.vae.decoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)

        self.predictor = nn.Sequential(
            nn.Linear(4096 , 64),
            nn.ReLU(),
            nn.Linear(64, 12) 
        )


    def forward(self, x):
        h = self.encoder(x)

        latent_space1 = self.conv1(h)
        mu = latent_space1[:, :1, :, :]
        log_var = latent_space1[:, 1:,:,:]



        std = log_var.mul(0.5).exp_()

        # sampling epsilon from normal distribution
        epsilon = torch.randn_like(std)
        latent_space = mu+std*epsilon
        latent_space1_1 = latent_space.view(latent_space1.size(0), -1)
        socio = self.predictor(latent_space1_1)

        latent_space2 = self.conv2(latent_space)
        reconstruncted_img = self.decoder(latent_space2)
        return reconstruncted_img, socio, mu, log_var

class myLoss(nn.Module):
    def __init__(self, alpha):
        super(myLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex')  
        

    def forward(self, reconstructed_images, original_images, predicted_variables, true_variables, mu, logvar):
        # Reconstruction loss
        loss_images = self.l1_loss(reconstructed_images, original_images)

        # LPIPS loss between original and reconstructed images
        lpips_loss = self.lpips_loss(reconstructed_images, original_images).mean()

        # KL divergence
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) *0.01

        # Calculate absolute error for socioeconomic variables
        loss_variables = torch.mean(torch.abs(predicted_variables - true_variables))

        # Total loss
        total_loss = (1 - self.alpha) * (loss_images + kl_divergence + lpips_loss )+ self.alpha * loss_variables
        return loss_images, lpips_loss, kl_divergence, loss_variables, total_loss
    

#Dataloader

def filter_none(json_data):
    filtered_data = {}
    for key, value in json_data.items():
        filtered_value = {}
        for subkey, subvalue in value.items():
            if subvalue is not None:
                filtered_value[subkey] = subvalue
        if filtered_value:
            filtered_data[key] = filtered_value
    return filtered_data

with open('../ACS_results.json', 'r') as f:
    data_info = json.load(f)

filtered_data_info = filter_none(data_info)

class Img_SC_Dataset(Dataset):
    def __init__(self, image_dir, data_info, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data_info = data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        filename = list(self.data_info.keys())[idx]
        image_path = os.path.join(self.image_dir, filename)
        socioeconomic_data = self.data_info[filename]

    # Image

        image = Image.open(image_path)
        image_size = image.size
        image_channels = image.mode


        if self.transform:
            image = self.transform(image)

    # Socioeconomic data

        values = list(socioeconomic_data.values())

        socioeconomic_values = np.array(values, dtype=np.float32)
        min_val = np.min(socioeconomic_values)
        max_val = np.max(socioeconomic_values)
        socioeconomic_values = (socioeconomic_values - min_val) / (max_val - min_val + 1e-8)

        return image, socioeconomic_values

#Transform
transform = transforms.Compose([
     #transforms.Resize((256, 256)),
      transforms.ToTensor(),
])

dataset_train = Img_SC_Dataset(image_dir='./rgb', data_info=filtered_data_info, transform=transform)
dataloader = DataLoader(dataset_train, batch_size=2, shuffle=True)



##############
# delete alpha channel 
# folder_path = "./512"

# file_list = os.listdir(folder_path)


# for file_name in file_list:

#     file_path = os.path.join(folder_path, file_name)

#     if file_name.endswith(".tif"):

#         image = Image.open(file_path)

#         r, g, b, a = image.split()

#         rgb_image = Image.merge("RGB", (r, g, b))

#         new_file_name = file_name
#         new_file_path = os.path.join("../rgb", new_file_name)
#         rgb_image.save(new_file_path)

#################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(AutoencoderKL).to(device)

curr_epoch = 0
epochs = 120
learning_rate = 0.0005
alpha = 0.5  
criterion = myLoss(alpha=alpha).to(device)
optimizer_AE = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer_AE, mode='min', factor=0.5, patience=3, verbose=True)


# ckpt = torch.load('model_epoch_20.pth')
# model.load_state_dict = ckpt['model_state_dict']
# curr_epoch = ckpt['epoch']


# train
for epoch in range(curr_epoch, epochs):
    model.train()
   

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        
        inputs, true_variables = data
        inputs=inputs.to(device)
        true_variables=true_variables.to(device)

       
        optimizer_AE.zero_grad()
        
       
        reconstructed_images, predicted_variables, mu, logvar = model(inputs)
        
        loss_images, lpips_loss, kl_divergence, loss_variables, total_loss = criterion(reconstructed_images, inputs, predicted_variables, true_variables, mu, logvar)

        total_loss.backward()
        optimizer_AE.step()

        running_loss += total_loss.item()
        print("loss at ", i, "step:", total_loss.item(), loss_images.item(), lpips_loss.item(), kl_divergence.item(), loss_variables.item())
        # break
    scheduler.step(running_loss / len(dataloader))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader)}")
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_AE.state_dict(),
            'loss': running_loss / len(dataloader),
        }, f'./model_epoch_{epoch+1}.pth')
        # model.save_pretrained(f'model_epoch_{epoch+1}')
        print(f"Model saved at epoch {epoch+1}")


