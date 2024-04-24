import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.basic_cnn import DenoisingCNN
import wandb


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for _, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)  # Assuming we want to learn a denoising task
        loss.backward()
        optimizer.step()
        wandb.log({"training_loss": loss.item()})


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(val_loader):
            data = data.to(device)
            output = model(data)
            val_loss += criterion(output, data).item()
    val_loss /= len(val_loader)
    wandb.log({"validation_loss": val_loss})


def main():
    with wandb.init():
        config = wandb.config

        # Set up CUDA
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # CIFAR-10 Datasets
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        dataset1 = datasets.CIFAR10("data", train=True, download=True, transform=transform)

        train_loader = DataLoader(dataset1, batch_size=config["batch_size"], shuffle=True)

        # Model
        model = DenoisingCNN(config).to(device)

        # Optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=config["T_max"], eta_min=0.00001
        )

        # Training loop
        for _ in range(1, config["epochs"] + 1):
            train(model, device, train_loader, optimizer, criterion)
            validate(model, device, train_loader, criterion)
            scheduler.step()


if __name__ == "__main__":
    main()
