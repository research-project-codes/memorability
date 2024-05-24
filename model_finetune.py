import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import lpips
from torchvision.models import vgg19
import time
from imagenet_autoencoder_main.models import vgg
from pytorch_msssim import SSIM, ms_ssim

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Define paths for the dataset
path = os.getcwd()
data_dir = os.path.join(path, 'data_MemCat')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Image transformations
train_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load the data
train_data = ImageFolder(train_dir, transform=train_preprocess)
val_data = ImageFolder(val_dir, transform=test_preprocess)
test_data = ImageFolder(test_dir, transform=test_preprocess)

# Load pretrained model
state_dict_path = os.path.join(path, 'imagenet_vgg16_autoencoder.pth')
state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
configs = vgg.get_configs()
model = vgg.VGGAutoEncoder(configs)
model.load_state_dict(state_dict, strict=False)
model.to(device)

# Define the Style Loss function
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.feature_extractor = vgg19(pretrained=True).features.eval()
        self.gram = lambda x: torch.mm(x, x.t())

    def forward(self, input, target):
        features_input = self.feature_extractor(input)
        features_target = self.feature_extractor(target)
        G_input = self.gram(features_input.view(features_input.size(1), -1))
        G_target = self.gram(features_target.view(features_target.size(1), -1))
        loss = nn.MSELoss()(G_input, G_target)
        return loss

def get_criterion(loss_fcn):
    if loss_fcn == 'ssim':
        return SSIM(data_range=1.0, channel=3, size_average=True).to(device)
    elif loss_fcn == 'lpips-vgg':
        return lpips.LPIPS(net='vgg').to(device)
    elif loss_fcn == 'lpips-sq':
        return lpips.LPIPS(net='squeeze').to(device)
    elif loss_fcn == 'lpips-alex':
        return lpips.LPIPS(net='alex').to(device)
    elif loss_fcn == 'styleLoss':
        return StyleLoss().to(device)
    elif loss_fcn == 'mae':
        return nn.L1Loss().to(device)
    elif loss_fcn == 'mse':
        return nn.MSELoss().to(device)
    elif loss_fcn == 'ms-ssim':
        return ms_ssim

def train_model(model, train_loader, criterion, optimizer, loss_fcn):
    model.train()
    train_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        reconstructed = model(images)
        if loss_fcn == 'ms-ssim':
            loss = 1 - criterion(images, reconstructed, data_range=1.0, size_average=True)
        elif isinstance(criterion, (SSIM, lpips.LPIPS)):
            loss = 1 - criterion(images, reconstructed)
        else:
            loss = criterion(reconstructed, images)
        if len(loss.size()) > 0:
            loss = loss.mean()
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    return train_loss

def validate_model(model, val_loader, criterion, loss_fcn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            if loss_fcn == 'ms-ssim':
                loss = 1 - criterion(images, outputs, data_range=1.0, size_average=True)
            elif isinstance(criterion, (SSIM, lpips.LPIPS)):
                loss = 1 - criterion(images, outputs)
            else:
                loss = criterion(outputs, images)
            if len(loss.size()) > 0:
                loss = loss.mean()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

def save_model_state(model, path, model_name):
    model_state_path = os.path.join(path, f'{model_name}_statedict.pth')
    torch.save(model.state_dict(), model_state_path)

# Define learning rates, batch sizes, and loss functions
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1]
batch_sizes = [1]
loss_funcs = ['lpips-sq', 'lpips-vgg', 'lpips-alex', 'mse', 'ssim', 'ms-ssim', 'mae', 'styleLoss']
num_epochs = 1

# Train the model
for loss_fcn in loss_funcs:
    criterion = get_criterion(loss_fcn)
    for lr in learning_rates:
        for bs in batch_sizes:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
            val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
            
            for epoch in range(num_epochs):
                model_name = f'MemCat_Autoencoder_vgg16_lr{lr}_bs{bs}_LossFunc{loss_fcn}_epoch{epoch+1}'
                start_time = time.time()
                train_loss = train_model(model, train_loader, criterion, optimizer, loss_fcn)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
                
                val_loss = validate_model(model, val_loader, criterion, loss_fcn)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}')
                
                save_model_state(model, path, model_name)
