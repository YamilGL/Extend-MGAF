# 6_train_inertial_cnn_cv.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# --- Parámetros de Configuración (Coinciden con el paper de inerciales) ---
# Directorio base que contiene las carpetas train/validation/test
BASE_DATA_DIR = 'datasets/Split_Data_Inertial' # <<< CAMBIO: Apunta a la nueva estructura de datos
# Ruta para guardar el modelo final entrenado con todos los datos
MODEL_SAVE_PATH = 'models/inertial_cnn_cv_final.pth'
NUM_CLASSES = 27
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.005
L2_REG = 0.004
MOMENTUM = 0.9
RANDOM_SEED = 42

# --- Parámetros de Validación Cruzada ---
K_FOLDS = 5 # Número de divisiones para la validación cruzada

# --- Dataset personalizado (Genérico, funciona para cualquier estructura de ImageFolder) ---
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if class_to_idx is None:
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        classes = self.class_to_idx.keys()

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_name]
            # Busca cualquier archivo de imagen común
            for img_path in glob.glob(os.path.join(class_dir, '*.png')):
                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Asegurar un solo canal (escala de grises)
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Modelo CNN (del paper, sin cambios) ---
class BaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaseCNN, self).__init__()
        self.stream = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.classifier = nn.Sequential(
             nn.Flatten(),
             nn.Linear(32 * 11 * 11, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.stream(x))

# --- Funciones auxiliares de entrenamiento y validación (sin cambios) ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total


# --- Script Principal ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    # 1. Definir transformaciones (simples, adecuadas para datos inerciales)
    transform = transforms.Compose([
        transforms.ToTensor(), # CustomImageDataset ya convierte a escala de grises
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Cargar los datos desde la nueva estructura de carpetas
    train_dir = os.path.join(BASE_DATA_DIR, 'train')
    val_dir = os.path.join(BASE_DATA_DIR, 'validation')
    test_dir = os.path.join(BASE_DATA_DIR, 'test')

    if not os.path.exists(train_dir):
        print(f"❌ ERROR: El directorio de entrenamiento '{train_dir}' no existe.")
        print("Asegúrate de haber ejecutado primero el script '3_split_dataset.py' para los datos inerciales.")
        return

    # Crear un mapeo de clases maestro para asegurar consistencia
    master_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    master_class_map = {cls_name: i for i, cls_name in enumerate(master_classes)}
    print(f"🗺️  Mapeo de {len(master_class_map)} clases creado.")

    # Crear datasets para train y validation
    train_dataset = CustomImageDataset(train_dir, transform=transform, class_to_idx=master_class_map)
    val_dataset = CustomImageDataset(val_dir, transform=transform, class_to_idx=master_class_map)
    
    # Combinar train y validation en un solo dataset para K-Fold
    full_train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"📊 Dataset combinado para CV: {len(full_train_val_dataset)} imágenes.")
    
    # El conjunto de Test se mantiene separado
    test_dataset = CustomImageDataset(test_dir, transform=transform, class_to_idx=master_class_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"🔒 Dataset de prueba (test) separado: {len(test_dataset)} imágenes.")

    # 3. Iniciar la Validación Cruzada (K-Fold)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    print(f"\n🔄 Iniciando Validación Cruzada de {K_FOLDS} pliegues (folds)...")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_val_dataset)):
        print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

        # Re-inicializar modelo y optimizador para cada fold
        model = BaseCNN(num_classes=NUM_CLASSES).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
        criterion = nn.CrossEntropyLoss()

        best_fold_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            print(f"  Época {epoch+1:02d}/{EPOCHS} -> "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
        
        print(f"✔️ Mejor Acc de validación para el Fold {fold+1}: {best_fold_acc:.2f}%")
        fold_results.append(best_fold_acc)

    # 4. Mostrar resultados de la Validación Cruzada
    print(f"\n{'='*20} RESULTADOS DE VALIDACIÓN CRUZADA {'='*20}")
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"📈 Precisión promedio en los {K_FOLDS} pliegues: {avg_acc:.2f}%")
    print(f"📊 Desviación estándar: {std_acc:.2f}%")

    # 5. Entrenamiento Final sobre TODOS los datos de train+validation
    print(f"\n{'='*20} ENTRENAMIENTO FINAL {'='*20}")
    print("Entrenando el modelo final con todos los datos de train y validation...")
    
    final_train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = BaseCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(final_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(EPOCHS), desc="Entrenamiento final"):
        train_loss, train_acc = train_epoch(final_model, final_train_loader, criterion, optimizer, device)

    print("✅ Entrenamiento final completado.")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Modelo final guardado en: {MODEL_SAVE_PATH}")

    # 6. Evaluación Final sobre el conjunto de TEST
    print(f"\n{'='*20} EVALUACIÓN FINAL EN EL CONJUNTO DE TEST {'='*20}")
    test_loss, test_acc = validate_epoch(final_model, test_loader, criterion, device)
    print(f"🏆 Rendimiento final en el conjunto de prueba (Test Set):")
    print(f"   - Pérdida (Loss): {test_loss:.4f}")
    print(f"   - Precisión (Accuracy): {test_acc:.2f}%")


if __name__ == '__main__':
    # Antes de ejecutar, asegúrate de que el directorio BASE_DATA_DIR existe
    # y contiene las carpetas 'train', 'validation' y 'test'
    if not os.path.isdir(BASE_DATA_DIR):
        print(f"Error: El directorio base '{BASE_DATA_DIR}' no fue encontrado.")
        print("Asegúrate de ejecutar el script '3_split_dataset.py' en los datos inerciales primero.")
    else:
        main()