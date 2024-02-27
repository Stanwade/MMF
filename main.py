import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets import MMFDataset
from networks import FullyConnectedNetwork, OneLayer
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load the dataset
train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=True)
valid_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)
print(f'Train dataset size: {len(train_dataset)}')
print(f'Valid dataset size: {len(valid_dataset)}')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
print(f'Train loader size: {len(train_loader)}')
print(f'Valid loader size: {len(valid_loader)}')


class TrainLoop:
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, 
                 valid_loader: DataLoader, 
                 criterion: nn.Module, 
                 optimizer: optim.Optimizer, 
                 device: torch.device):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(self.valid_loader.dataset)
        return epoch_loss

    def run(self, num_epochs):
        print(f'Running {num_epochs} epochs...')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            valid_loss = self.validate_epoch()
            # progress_bar = tqdm(range(num_epochs), desc='Training Progress')
            # for epoch in progress_bar:
            #     train_loss = self.train_epoch()
            #     valid_loss = self.validate_epoch()
                
            #     progress_bar.set_postfix({'Train Loss': train_loss, 'Valid Loss': valid_loss})
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            # save the model every 100 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f'model_checkpoint_{epoch+1}.pth')
            
            # early stopping
            if valid_loss < 0.01:
                print('Early stopping')
                torch.save(self.model.state_dict(), f'early_stop_{epoch+1}.pth')
                break

# Create the model
model = FullyConnectedNetwork().to(device)
print('model created')

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create the training loop
train_loop = TrainLoop(model, train_loader, valid_loader, criterion, optimizer, device)

if __name__ == '__main__':
    train_loop.run(100)
