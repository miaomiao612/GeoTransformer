import json

json_file_path = 'GDP_results.json'
# json_file_path = 'Trips_results.json'

with open(json_file_path, 'r') as file:
    gdp_data = json.load(file)
# gdp_data = {k: v for k, v in gdp_data.items() if v['Trips'] is not None}

import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PIL import Image
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GDPDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = list(self.data.keys())[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        gdp = torch.tensor([self.data[img_name]['F2019GDP']])
        # gdp = torch.tensor([self.data[img_name]['Trips']])

        if self.transform:
            image = self.transform(image)

        return image, gdp

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = GDPDataset(gdp_data, 'rgb/', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(len(train_loader))

model = ViTModel.from_pretrained('google/vit-base-patch16-224')

for param in model.parameters():
    param.requires_grad = False

num_layers = len(model.encoder.layer)
layers_to_unfreeze = 4
model.classifier = torch.nn.Linear(model.config.hidden_size, 1)

for layer in model.encoder.layer[-layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True
model.classifier = torch.nn.Linear(model.config.hidden_size, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

def train(epoch):
    model.train()
    train_loss, train_mae, train_r2 = [], [], []
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}/Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).last_hidden_state[:, 0]
        outputs = model.classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        predictions = outputs.detach().cpu()
        labels = labels.cpu()
        train_mae.append(mean_absolute_error(labels, predictions))
        train_r2.append(r2_score(labels, predictions))

    avg_loss = torch.mean(torch.tensor(train_loss))
    avg_mae = torch.mean(torch.tensor(train_mae))
    avg_r2 = torch.mean(torch.tensor(train_r2))
    print(f"Epoch {epoch}: Average Train Loss = {avg_loss:.4f}, Average MAE = {avg_mae:.4f}, Average R2 = {avg_r2:.4f}")

def test(epoch):
    model.eval()
    test_loss, test_mae, test_r2 = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Epoch {epoch}/Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).last_hidden_state[:, 0]
            outputs = model.classifier(outputs)
            loss = criterion(outputs, labels)

            test_loss.append(loss.item())
            predictions = outputs.cpu()
            labels = labels.cpu()
            test_mae.append(mean_absolute_error(labels, predictions))
            test_r2.append(r2_score(labels, predictions))

    avg_loss = torch.mean(torch.tensor(test_loss))
    avg_mae = torch.mean(torch.tensor(test_mae))
    avg_r2 = torch.mean(torch.tensor(test_r2))
    print(f"Epoch {epoch}: Average Test Loss = {avg_loss:.4f}, Average MAE = {avg_mae:.4f}, Average R2 = {avg_r2:.4f}")
    scheduler.step(avg_loss)

num_epochs = 50
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)


torch.save(model.state_dict(), 'baseline_vit.pth')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}, 'model_vit_full.pth')