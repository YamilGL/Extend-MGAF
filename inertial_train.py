import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# Modelo XONet2
class XONet2(nn.Module):
    def __init__(self, num_classes=10):
        super(XONet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)     # -> 16x60x60
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 16x30x30
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)    # -> 32x26x26
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)    # -> 32x22x22
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32x11x11

        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> 16x60x60
        x = self.pool1(x)           # -> 16x30x30
        x = F.relu(self.conv2(x))   # -> 32x26x26
        x = F.relu(self.conv3(x))   # -> 32x22x22
        x = self.pool2(x)           # -> 32x11x11
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        return x

# Parámetros (del paper)
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.004
DATA_DIR = 'dataset/inertialimages_64x64'

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Cargar dataset con subcarpetas por clase
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inicializar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XONet2(num_classes=num_classes).to(device)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY)

# Entrenamiento
print("🧠 Entrenando XONet2...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Estadísticas
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Época {epoch+1}/{EPOCHS}, Pérdida: {running_loss:.4f}, Precisión: {acc:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), 'XONet_InertialImages_64x64.pth')
print("✅ Modelo XONet2 guardado como 'XONet_InertialImages_64x64.pth'")
