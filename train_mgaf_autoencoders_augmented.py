# 7_train_mgaf_cv.py (ARQUITECTURA CON ENCODER POST-FUSIÓN)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import torchvision.transforms as transforms

# --- Parámetros de Configuración ---
BASE_DEPTH_DIR = 'datasets/Split_Data_Depth'
BASE_INERTIAL_DIR = 'datasets/Split_Data_Inertial'
PRETRAINED_DEPTH_MODEL = 'models/depth_cnn_cv_final.pth'
PRETRAINED_INERTIAL_MODEL = 'models/inertial_cnn_cv_final.pth'
MGAF_SAVE_PATH = 'models/mgaf_model_cv_final_post_fusion_encoder.pth' # Nuevo nombre

NUM_CLASSES = 27
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
K_FOLDS = 5

# --- Dataset Personalizado (Sin Cambios) ---
class PairedDataset(Dataset):
    def __init__(self, depth_split_dir, inertial_split_dir, transform=None, class_to_idx=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        
        print(f"\n[Dataset] Buscando pares en (Profundidad: '{depth_split_dir}', Inercial: '{inertial_split_dir}')")
        
        found_count = 0
        if not os.path.exists(depth_split_dir):
            print(f"  [ERROR] El directorio de profundidad no existe: {depth_split_dir}")
            return

        for class_name in sorted(os.listdir(depth_split_dir)):
            class_path_depth = os.path.join(depth_split_dir, class_name)
            class_path_inertial = os.path.join(inertial_split_dir, class_name)

            if not os.path.isdir(class_path_depth) or not os.path.isdir(class_path_inertial):
                continue
                
            for depth_path in glob.glob(os.path.join(class_path_depth, '*_sfi_frame*.png')):
                base_name = os.path.basename(depth_path)
                sample_base = '_'.join(base_name.split('_')[:3])
                inertial_pattern = os.path.join(class_path_inertial, f"{sample_base}_inertial*.png")
                matching_inertial_files = glob.glob(inertial_pattern)
                
                for inertial_path in matching_inertial_files:
                    label = self.class_to_idx[class_name]
                    self.samples.append((depth_path, inertial_path, label))
                    found_count += 1
        
        print(f"[Dataset] Resumen de búsqueda:")
        print(f"  - Pares de imágenes (depth-inertial) creados: {found_count}")
        if found_count == 0:
            print("  [CRÍTICO] No se creó NINGÚN par. Revisa los patrones de nombres de archivo y las rutas.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        depth_path, inertial_path, label = self.samples[idx]
        depth_img = Image.open(depth_path).convert('L')
        inertial_img = Image.open(inertial_path).convert('L')
        if self.transform:
            depth_img = self.transform(depth_img)
            inertial_img = self.transform(inertial_img)
        return depth_img, inertial_img, label


# --- Bloque GAF (Sin Cambios) ---
class GAF_Block(nn.Module):
    def __init__(self, channels):
        super(GAF_Block, self).__init__()
        self.kernel_depth = nn.Conv2d(channels, channels, kernel_size=1)
        self.kernel_inertial = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_depth, f_inertial):
        gate_depth = self.sigmoid(self.kernel_depth(f_depth))
        gate_inertial = self.sigmoid(self.kernel_inertial(f_inertial))
        gated_depth = f_depth * gate_depth
        gated_inertial = f_inertial * gate_inertial
        return gated_depth + gated_inertial

# --- Bloque Encoder (Sin Cambios) ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.encoder(x)

# =========================================================================
# === INICIO DE LAS MODIFICACIONES DE ARQUITECTURA ===
# =========================================================================
class MGAFModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(MGAFModel, self).__init__()
        
        # Parámetros de canales
        stream_out_channels = 32
        bottleneck_channels = 16 # Dimensionalidad del espacio latente comprimido

        # 1. Flujos de extracción de características
        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, stream_out_channels, kernel_size=5), nn.BatchNorm2d(stream_out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(stream_out_channels, stream_out_channels, kernel_size=5), nn.BatchNorm2d(stream_out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        self.inertial_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, stream_out_channels, kernel_size=5), nn.ReLU(),
            nn.Conv2d(stream_out_channels, stream_out_channels, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. Módulo de fusión GAF
        # Opera sobre los canales de salida de los streams (32)
        self.gaf_fusion = GAF_Block(channels=stream_out_channels)
        
        # 3. [NUEVO] Encoder único POST-fusión
        # Comprime el mapa de características fusionado de 32 a 16 canales.
        self.post_fusion_encoder = EncoderBlock(stream_out_channels, bottleneck_channels)
        
        # 4. [MODIFICADO] Clasificador
        # La entrada al clasificador es la salida del encoder.
        # Tamaño aplanado = 16 * 11 * 11 = 1936
        final_feature_dim = bottleneck_channels * 11 * 11
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_feature_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x_depth, x_inertial):
        # 1. Extraer características
        f_depth = self.depth_stream(x_depth)       # Salida: [B, 32, 11, 11]
        f_inertial = self.inertial_stream(x_inertial) # Salida: [B, 32, 11, 11]
        
        # 2. Fusionar las características
        fused = self.gaf_fusion(f_depth, f_inertial) # Salida: [B, 32, 11, 11]
        
        # 3. Comprimir/Codificar el mapa de características fusionado
        encoded_fused = self.post_fusion_encoder(fused) # Salida: [B, 16, 11, 11]
        
        # 4. Clasificar el resultado
        output = self.classifier(encoded_fused)
        return output

# =========================================================================
# === FIN DE LAS MODIFICACIONES DE ARQUITECTURA ===
# =========================================================================

def load_pretrained_weights(model, depth_path, inertial_path, device):
    print("🔄 Cargando pesos pre-entrenados...")
    try:
        depth_state_dict = torch.load(depth_path, map_location=device)
        inertial_state_dict = torch.load(inertial_path, map_location=device)
    except FileNotFoundError as e:
        print(f"❌ ERROR: No se encontró el archivo del modelo: {e}")
        exit(1)
    model_dict = model.state_dict()
    depth_weights_to_load = {k: v for k, v in depth_state_dict.items() if k.startswith('features.')}
    inertial_weights_to_load = {k: v for k, v in inertial_state_dict.items() if k.startswith('stream.')}
    renamed_depth_weights = {f"depth_stream.{k[9:]}": v for k, v in depth_weights_to_load.items()}
    renamed_inertial_weights = {f"inertial_stream.{k[7:]}": v for k, v in inertial_weights_to_load.items()}
    model_dict.update(renamed_depth_weights)
    model.load_state_dict(model_dict)
    print("✅ Pesos de los flujos de profundidad e inercial cargados exitosamente.")
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for d_img, i_img, labels in dataloader:
        d_img, i_img, labels = d_img.to(device), i_img.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(d_img, i_img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * d_img.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for d_img, i_img, labels in dataloader:
            d_img, i_img, labels = d_img.to(device), i_img.to(device), labels.to(device)
            outputs = model(d_img, i_img)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * d_img.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    master_class_dir = os.path.join(BASE_DEPTH_DIR, 'train')
    master_classes = sorted([d for d in os.listdir(master_class_dir) if os.path.isdir(os.path.join(master_class_dir, d))])
    master_class_map = {cls_name: i for i, cls_name in enumerate(master_classes)}
    print(f"🗺️  Mapeo de {len(master_class_map)} clases creado.")
    train_dataset = PairedDataset(os.path.join(BASE_DEPTH_DIR, 'train'), os.path.join(BASE_INERTIAL_DIR, 'train'), transform, master_class_map)
    val_dataset = PairedDataset(os.path.join(BASE_DEPTH_DIR, 'validation'), os.path.join(BASE_INERTIAL_DIR, 'validation'), transform, master_class_map)
    full_train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"📊 Dataset combinado para CV: {len(full_train_val_dataset)} pares de imágenes.")
    if len(full_train_val_dataset) == 0:
        print("\n[ERROR FATAL] El dataset combinado está vacío. El script no puede continuar.")
        return
    test_dataset = PairedDataset(os.path.join(BASE_DEPTH_DIR, 'test'), os.path.join(BASE_INERTIAL_DIR, 'test'), transform, master_class_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"🔒 Dataset de prueba (test) separado: {len(test_dataset)} pares de imágenes.")
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    print(f"\n🔄 Iniciando Validación Cruzada de {K_FOLDS} pliegues...")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_val_dataset)):
        print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")
        train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_ids), num_workers=2, pin_memory=True)
        val_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_ids), num_workers=2, pin_memory=True)
        model = MGAFModel(num_classes=NUM_CLASSES).to(device)
        model = load_pretrained_weights(model, PRETRAINED_DEPTH_MODEL, PRETRAINED_INERTIAL_MODEL, device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        best_fold_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            print(f"  Época {epoch+1:02d}/{EPOCHS} -> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            if val_acc > best_fold_acc: best_fold_acc = val_acc
        print(f"✔️ Mejor Acc de validación para el Fold {fold+1}: {best_fold_acc:.2f}%")
        fold_results.append(best_fold_acc)
    print(f"\n{'='*20} RESULTADOS DE VALIDACIÓN CRUZADA {'='*20}")
    print(f"📈 Precisión promedio en los {K_FOLDS} pliegues: {np.mean(fold_results):.2f}% (±{np.std(fold_results):.2f}%)")
    print(f"\n{'='*20} ENTRENAMIENTO FINAL {'='*20}")
    final_train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    final_model = MGAFModel(num_classes=NUM_CLASSES).to(device)
    final_model = load_pretrained_weights(final_model, PRETRAINED_DEPTH_MODEL, PRETRAINED_INERTIAL_MODEL, device)
    optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(EPOCHS), desc="Entrenamiento final del modelo MGAF"):
        train_epoch(final_model, final_train_loader, criterion, optimizer, device)
    print("✅ Entrenamiento final completado.")
    os.makedirs(os.path.dirname(MGAF_SAVE_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MGAF_SAVE_PATH)
    print(f"💾 Modelo final MGAF guardado en: {MGAF_SAVE_PATH}")
    print(f"\n{'='*20} EVALUACIÓN FINAL EN EL CONJUNTO DE TEST {'='*20}")
    test_loss, test_acc = validate_epoch(final_model, test_loader, criterion, device)
    print(f"🏆 Rendimiento final en el conjunto de prueba (Test Set):")
    print(f"   - Pérdida (Loss): {test_loss:.4f}")
    print(f"   - Precisión (Accuracy): {test_acc:.2f}%")


if __name__ == '__main__':
    main()