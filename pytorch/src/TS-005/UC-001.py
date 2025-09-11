import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from  gt.core import Executable
import gguf
from gguf import GGUFWriter
import os


@Executable("train and save MLP MNIST")
def train_and_save_mnist():
    
    train_and_save()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)  # random tensor to keep up with API
    return [x], x


def train_and_save():
    # Define the PyTorch model
    class MNIST_MLP(nn.Module):
        def __init__(self):
            super(MNIST_MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.model(x)

    # Prepare dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNIST_MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # Save the trained PyTorch model
    torch.save(model.state_dict(), 'mnist_mlp.pth')

    # Export to GGUF format
    # Install the necessary package: pip install gguf

    model_cpu = MNIST_MLP().to('cpu')
    model_cpu.load_state_dict(torch.load('mnist_mlp.pth'))
    model_cpu.eval()
    
    file_name = "mnist_mlp.gguf"
    full_path = os.path.abspath(file_name)
    print(f"Full path to GGUF file: {full_path}")

    writer = GGUFWriter(full_path, arch="generic")
    for name, param in model_cpu.named_parameters():
        writer.add_tensor(name, param.detach().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


