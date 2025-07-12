# 5_train_depth_cnn_cv.py
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

# --- Parámetros de Configuración ---
# Directorio base que contiene las carpetas train/validation/test
BASE_DATA_DIR = 'datasets/Split_Data_Depth'
# Ruta para guardar el modelo final entrenado con todos los datos
MODEL_SAVE_PATH = 'models/depth_cnn_cv_final.pth'
NUM_CLASSES = 27
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.01
L2_REG = 0.001
MOMENTUM = 0.9
SCHEDULER_STEP = 10
SCHEDULER_GAMMA = 0.5
RANDOM_SEED = 42

# --- Parámetros de Validación Cruzada ---
K_FOLDS = 5 # Número de divisiones para la validación cruzada

# --- Dataset personalizado (Modificado para aceptar un mapeo de clases) ---
class DepthImageDataset(Dataset):
    def __init__(self, root_dir, image_type='sfi', transform=None, class_to_idx=None):
        """
        image_type: 'sfi', 'mei', 'mhi', o 'all'
        class_to_idx: Un diccionario predefinido para mapear nombres de clase a índices.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_type = image_type
        self.samples = []
        
        # Si no se proporciona un mapeo, créalo. De lo contrario, úsalo.
        if class_to_idx is None:
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        classes = self.class_to_idx.keys()

        # Recopilar muestras
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue # Saltar si una clase no está presente en este split
            
            class_idx = self.class_to_idx[class_name]
            
            if image_type == 'sfi':
                pattern = os.path.join(class_dir, "*_sfi_frame*.png")
            # Agrega otras condiciones si necesitas usar 'mei' o 'mhi'
            else:  # 'all'
                pattern = os.path.join(class_dir, "*.png")
            
            files = glob.glob(pattern)
            for file_path in files:
                self.samples.append((file_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Modelo CNN (sin cambios) ---
class ImprovedPaperCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedPaperCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(self.features(x))

# --- Función para entrenar un ciclo (época) ---
def train_epoch(model, dataloader, criterion, optimizer, device):
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
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# --- Función para validar/evaluar un ciclo (época) ---
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# --- Script Principal ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    # 1. Definir transformaciones
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Cargar los datos desde la nueva estructura de carpetas
    train_dir = os.path.join(BASE_DATA_DIR, 'train')
    val_dir = os.path.join(BASE_DATA_DIR, 'validation')
    test_dir = os.path.join(BASE_DATA_DIR, 'test')

    if not os.path.exists(train_dir):
        print(f"❌ ERROR: El directorio de entrenamiento '{train_dir}' no existe.")
        print("Asegúrate de haber ejecutado primero el script '3_split_dataset.py'.")
        return

    # Crear un mapeo de clases maestro desde el directorio de entrenamiento para asegurar consistencia
    master_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    master_class_map = {cls_name: i for i, cls_name in enumerate(master_classes)}
    print(f"🗺️  Mapeo de {len(master_class_map)} clases creado.")

    # Crear datasets para train y validation (usando solo SFI por ahora)
    # Estos se combinarán para la validación cruzada
    train_dataset = DepthImageDataset(train_dir, image_type='sfi', transform=train_transform, class_to_idx=master_class_map)
    val_dataset = DepthImageDataset(val_dir, image_type='sfi', transform=train_transform, class_to_idx=master_class_map) # Usamos train_transform
    
    # Combinar train y validation en un solo dataset para K-Fold
    full_train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"📊 Dataset combinado para CV: {len(full_train_val_dataset)} imágenes.")
    
    # El conjunto de Test se mantiene separado con transformaciones de validación
    test_dataset = DepthImageDataset(test_dir, image_type='sfi', transform=val_transform, class_to_idx=master_class_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"🔒 Dataset de prueba (test) separado: {len(test_dataset)} imágenes.")

    # 3. Iniciar la Validación Cruzada (K-Fold)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    print(f"\n🔄 Iniciando Validación Cruzada de {K_FOLDS} pliegues (folds)...")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_val_dataset)):
        print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")

        # Crear samplers para obtener los splits de este fold
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # Crear DataLoaders para este fold
        train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

        # Re-inicializar el modelo y optimizador para cada fold
        model = ImprovedPaperCNN(num_classes=NUM_CLASSES).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
        criterion = nn.CrossEntropyLoss()

        best_fold_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()
            
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
    
    # Usar el dataset combinado completo
    final_train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Re-inicializar modelo y optimizador una última vez
    final_model = ImprovedPaperCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(final_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(EPOCHS), desc="Entrenamiento final"):
        train_loss, train_acc = train_epoch(final_model, final_train_loader, criterion, optimizer, device)
        scheduler.step()

    print("✅ Entrenamiento final completado.")
    # Guardar el modelo final
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Modelo final guardado en: {MODEL_SAVE_PATH}")

    # 6. Evaluación Final sobre el conjunto de TEST (intacto)
    print(f"\n{'='*20} EVALUACIÓN FINAL EN EL CONJUNTO DE TEST {'='*20}")
    test_loss, test_acc = validate_epoch(final_model, test_loader, criterion, device)
    print(f"🏆 Rendimiento final en el conjunto de prueba (Test Set):")
    print(f"   - Pérdida (Loss): {test_loss:.4f}")
    print(f"   - Precisión (Accuracy): {test_acc:.2f}%")


if __name__ == '__main__':
    main()