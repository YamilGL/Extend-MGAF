import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.special import expit

# ---------------------------- #
#   MÓDULO DE ATENCIÓN CRUZADA
# ---------------------------- #

class CrossAttention(nn.Module):
    """
    Módulo de atención cruzada que permite que una modalidad atienda a otra.
    """
    def __init__(self, feature_dim, attention_dim=64):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Proyecciones lineales para Q, K, V
        self.query_proj = nn.Linear(feature_dim, attention_dim)
        self.key_proj = nn.Linear(feature_dim, attention_dim)
        self.value_proj = nn.Linear(feature_dim, attention_dim)
        
        # Proyección de salida
        self.output_proj = nn.Linear(attention_dim, feature_dim)
        
        # Factor de escala para la atención
        self.scale = attention_dim ** -0.5
        
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: Características de la modalidad que realiza la consulta (B, D)
            key_value_features: Características de la modalidad atendida (B, D)
        Returns:
            attended_features: Características atendidas (B, D)
        """
        batch_size = query_features.size(0)
        
        # Generar Q, K, V
        Q = self.query_proj(query_features)  # (B, attention_dim)
        K = self.key_proj(key_value_features)  # (B, attention_dim)
        V = self.value_proj(key_value_features)  # (B, attention_dim)
        
        # Calcular scores de atención
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, B)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Aplicar atención a los valores
        attended = torch.matmul(attention_weights, V)  # (B, attention_dim)
        
        # Proyección final
        output = self.output_proj(attended)  # (B, feature_dim)
        
        return output, attention_weights

def cross_attention_gated_fusion(F1, F2, use_cuda=False):
    """
    Fusión con atención cruzada y gating mejorado.
    
    Args:
        F1: Características de la primera modalidad (numpy array)
        F2: Características de la segunda modalidad (numpy array)
        use_cuda: Si usar GPU para los cálculos de atención
    
    Returns:
        F_fused: Características fusionadas
    """
    # Ajustar dimensiones para que coincidan
    if F1.shape[0] != F2.shape[0]:
        min_dim = min(F1.shape[0], F2.shape[0])
        F1 = F1[:min_dim]
        F2 = F2[:min_dim]
    
    # Convertir a tensores de PyTorch
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    F1_tensor = torch.FloatTensor(F1).to(device)
    F2_tensor = torch.FloatTensor(F2).to(device)
    
    feature_dim = F1_tensor.shape[1]
    
    # Crear módulos de atención cruzada
    cross_attention_1to2 = CrossAttention(feature_dim).to(device)
    cross_attention_2to1 = CrossAttention(feature_dim).to(device)
    
    with torch.no_grad():
        # F1 atiende a F2
        F1_attended, attn_weights_1to2 = cross_attention_1to2(F1_tensor, F2_tensor)
        
        # F2 atiende a F1  
        F2_attended, attn_weights_2to1 = cross_attention_2to1(F2_tensor, F1_tensor)
        
        # Combinar características originales con las atendidas
        F1_enhanced = F1_tensor + 0.5 * F1_attended  # Conexión residual
        F2_enhanced = F2_tensor + 0.5 * F2_attended
        
        # Calcular pesos de gating mejorados usando características atendidas
        # Usar tanto características originales como atendidas para el gating
        gate_input_1 = torch.cat([F1_tensor, F1_attended], dim=1)
        gate_input_2 = torch.cat([F2_tensor, F2_attended], dim=1)
        
        # Proyecciones para reducir dimensionalidad del gating
        gate_proj_1 = nn.Linear(feature_dim * 2, feature_dim).to(device)
        gate_proj_2 = nn.Linear(feature_dim * 2, feature_dim).to(device)
        
        with torch.no_grad():
            gate_1 = torch.sigmoid(gate_proj_1(gate_input_1))
            gate_2 = torch.sigmoid(gate_proj_2(gate_input_2))
        
        # Aplicar gating a las características mejoradas
        F1_gated = gate_1 * F1_enhanced
        F2_gated = gate_2 * F2_enhanced
        
        # Fusión final
        F_fused = F1_gated + F2_gated
        
        # Convertir de vuelta a numpy
        F_fused_np = F_fused.cpu().numpy()
        
    return F_fused_np

def gated_average_fusion(F1, F2):
    """
    Implementación original de GAF para comparación.
    """
    # Ajustar dimensiones para que coincidan
    if F1.shape[0] != F2.shape[0]:
        min_dim = min(F1.shape[0], F2.shape[0])
        F1 = F1[:min_dim]
        F2 = F2[:min_dim]
    
    # Calcular los pesos de las puertas utilizando la función sigmoide
    W1 = 1 / (1 + np.exp(-F1))  # Peso para F1
    W2 = 1 / (1 + np.exp(-F2))  # Peso para F2
    
    # Aplicar los pesos a las características
    F1_gated = W1 * F1
    F2_gated = W2 * F2
    
    # Fusión mediante suma
    F_fused = F1_gated + F2_gated
    
    return F_fused

def extract_features(model, dataset, layer_names, device='cpu'):
    """
    Extrae características de múltiples capas de un modelo CNN para un conjunto de datos.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features_dict = {layer: [] for layer in layer_names}
    labels = []
    
    # Registrar hooks para todas las capas solicitadas
    handles = []
    
    def get_hook_fn(layer_name):
        def hook_fn(module, input, output):
            # Para capas convolucionales, convertimos la salida a vector
            if isinstance(output, torch.Tensor) and len(output.shape) > 2:
                # Aplicar pooling global para reducir dimensiones espaciales
                out_pooled = F.adaptive_avg_pool2d(output, (1, 1))
                out_flat = out_pooled.view(out_pooled.size(0), -1)
                features_dict[layer_name].append(out_flat.detach().cpu().numpy())
            else:
                features_dict[layer_name].append(output.detach().cpu().numpy())
        return hook_fn
    
    # Registrar hooks para cada capa
    for layer_name in layer_names:
        layer = dict([*model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(get_hook_fn(layer_name))
        handles.append(handle)
    
    # Procesar el dataset
    for images, label in tqdm(loader, desc=f'Extracting features'):
        images = images.to(device)
        labels.extend(label.numpy().tolist())
        with torch.no_grad():
            model(images)
    
    # Eliminar los hooks
    for handle in handles:
        handle.remove()
    
    # Concatenar resultados
    for layer_name in layer_names:
        features_dict[layer_name] = np.concatenate(features_dict[layer_name], axis=0)
    
    return features_dict, np.array(labels)

# ---------------------------- #
#   DEFINICIÓN DE MODELOS
# ---------------------------- #

class XONet(nn.Module):
    """
    Implementación de la arquitectura XONet como se describe en el paper MGAF.
    """
    def __init__(self, num_classes=10):
        super(XONet, self).__init__()
        # Primera capa convolucional (Conv1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Segunda capa convolucional (Conv2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        
        # Tercera capa convolucional (Conv3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculamos el tamaño de salida después de las convoluciones y pooling
        flat_size = 32 * 11 * 11
        
        # Capa fully connected (FC1)
        self.fc1 = nn.Linear(flat_size, 128)
        
        # Capa de salida
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))   # Conv1
        x = self.pool1(x)           # Pool1
        x = F.relu(self.conv2(x))   # Conv2
        x = F.relu(self.conv3(x))   # Conv3
        x = self.pool2(x)           # Pool2
        x = x.view(x.size(0), -1)   # Aplanar
        x = F.relu(self.fc1(x))     # FC1
        x = self.fc_out(x)          # Salida
        return x

# ---------------------------- #
#   CARGA Y TRANSFORMACIONES
# ---------------------------- #

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Cargar datasets
depth_dataset = datasets.ImageFolder(root='dataset/DepthImages_64x64', transform=transform)
inertial_dataset = datasets.ImageFolder(root='dataset/InertialImages_64x64', transform=transform)

# Balanceo del dataset de profundidad
min_count = 2500
balanced_depth = []
for label in set(depth_dataset.targets):
    indices = [i for i, t in enumerate(depth_dataset.targets) if t == label]
    np.random.shuffle(indices)
    balanced_depth += indices[:min_count]

balanced_depth_dataset = torch.utils.data.Subset(depth_dataset, balanced_depth)

# División en train/test para ambos datasets
train_len_depth = int(0.8 * len(balanced_depth_dataset))
depth_train, depth_val = random_split(
    balanced_depth_dataset, 
    [train_len_depth, len(balanced_depth_dataset) - train_len_depth]
)

train_len_inertial = int(0.8 * len(inertial_dataset))
inertial_train, inertial_val = random_split(
    inertial_dataset, 
    [train_len_inertial, len(inertial_dataset) - train_len_inertial]
)

# ---------------------------- #
#   CARGA DE MODELOS ENTRENADOS
# ---------------------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NUM_CLASSES = 27

# Crear y cargar modelos para cada modalidad
depth_model = XONet(num_classes=NUM_CLASSES)
depth_model.load_state_dict(torch.load('XONet_DepthImages_64x64.pth', map_location=device))
depth_model.to(device).eval()

inertial_model = XONet(num_classes=NUM_CLASSES)
inertial_model.load_state_dict(torch.load('XONet_inertialimages_64x64.pth', map_location=device))
inertial_model.to(device).eval()

# ---------------------------- #
#   EXTRACCIÓN DE CARACTERÍSTICAS
# ---------------------------- #

layers = ['conv1', 'conv2', 'conv3', 'fc1']

# Extracción de características
print("Extrayendo características del modelo de profundidad...")
depth_features_train, y_train_depth = extract_features(depth_model, depth_train, layers, device)
depth_features_val, y_val_depth = extract_features(depth_model, depth_val, layers, device)

print("Extrayendo características del modelo inercial...")
inertial_features_train, y_train_inertial = extract_features(inertial_model, inertial_train, layers, device)
inertial_features_val, y_val_inertial = extract_features(inertial_model, inertial_val, layers, device)

# ---------------------------- #
#   COMPARACIÓN: GAF vs MGAF CON ATENCIÓN CRUZADA
# ---------------------------- #

print("Fusionando características con GAF tradicional...")

# GAF tradicional
fused_train_features_gaf = []
for layer in layers:
    fused_layer = gated_average_fusion(
        depth_features_train[layer], 
        inertial_features_train[layer]
    )
    fused_train_features_gaf.append(fused_layer)

fused_val_features_gaf = []
for layer in layers:
    fused_layer = gated_average_fusion(
        depth_features_val[layer], 
        inertial_features_val[layer]
    )
    fused_val_features_gaf.append(fused_layer)

X_train_gaf = np.concatenate(fused_train_features_gaf, axis=1)
X_val_gaf = np.concatenate(fused_val_features_gaf, axis=1)

print("Fusionando características con MGAF + Atención Cruzada...")

# MGAF con atención cruzada
use_gpu = torch.cuda.is_available()
fused_train_features_cross_attn = []
for layer in layers:
    fused_layer = cross_attention_gated_fusion(
        depth_features_train[layer], 
        inertial_features_train[layer],
        use_cuda=use_gpu
    )
    fused_train_features_cross_attn.append(fused_layer)

fused_val_features_cross_attn = []
for layer in layers:
    fused_layer = cross_attention_gated_fusion(
        depth_features_val[layer], 
        inertial_features_val[layer],
        use_cuda=use_gpu
    )
    fused_val_features_cross_attn.append(fused_layer)

X_train_cross_attn = np.concatenate(fused_train_features_cross_attn, axis=1)
X_val_cross_attn = np.concatenate(fused_val_features_cross_attn, axis=1)

y_train = y_train_inertial
y_val = y_val_inertial

print(f"Dimensión GAF tradicional - Train: {X_train_gaf.shape}, Val: {X_val_gaf.shape}")
print(f"Dimensión MGAF + Cross Attention - Train: {X_train_cross_attn.shape}, Val: {X_val_cross_attn.shape}")

# ---------------------------- #
#   ENTRENAMIENTO Y EVALUACIÓN SVM
# ---------------------------- #

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Evaluar GAF tradicional
print("Entrenando SVM con GAF tradicional...")
grid_search_gaf = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_gaf.fit(X_train_gaf, y_train)
best_svm_gaf = grid_search_gaf.best_estimator_

y_pred_gaf = best_svm_gaf.predict(X_val_gaf)
accuracy_gaf = accuracy_score(y_val, y_pred_gaf)

# Evaluar MGAF con atención cruzada
print("Entrenando SVM con MGAF + Atención Cruzada...")
grid_search_cross_attn = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_cross_attn.fit(X_train_cross_attn, y_train)
best_svm_cross_attn = grid_search_cross_attn.best_estimator_

y_pred_cross_attn = best_svm_cross_attn.predict(X_val_cross_attn)
accuracy_cross_attn = accuracy_score(y_val, y_pred_cross_attn)

# ---------------------------- #
#   RESULTADOS Y VISUALIZACIÓN
# ---------------------------- #

print("\n" + "="*50)
print("RESULTADOS COMPARATIVOS:")
print("="*50)
print(f"GAF Tradicional:")
print(f"  - Mejores parámetros: {grid_search_gaf.best_params_}")
print(f"  - Precisión: {accuracy_gaf:.4f}")
print(f"\nMGAF + Atención Cruzada:")
print(f"  - Mejores parámetros: {grid_search_cross_attn.best_params_}")  
print(f"  - Precisión: {accuracy_cross_attn:.4f}")
print(f"\nMejora: {((accuracy_cross_attn - accuracy_gaf) / accuracy_gaf * 100):.2f}%")

# Matrices de confusión comparativas
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# GAF tradicional
conf_mat_gaf = confusion_matrix(y_val, y_pred_gaf)
norm_conf_mat_gaf = conf_mat_gaf.astype('float') / conf_mat_gaf.sum(axis=1)[:, np.newaxis]

sns.heatmap(norm_conf_mat_gaf, annot=True, fmt=".2f", cmap='Blues', ax=axes[0])
axes[0].set_title(f'GAF Tradicional (Accuracy: {accuracy_gaf:.4f})')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Valor Real')

# MGAF con atención cruzada
conf_mat_cross_attn = confusion_matrix(y_val, y_pred_cross_attn)
norm_conf_mat_cross_attn = conf_mat_cross_attn.astype('float') / conf_mat_cross_attn.sum(axis=1)[:, np.newaxis]

sns.heatmap(norm_conf_mat_cross_attn, annot=True, fmt=".2f", cmap='Greens', ax=axes[1])
axes[1].set_title(f'MGAF + Cross Attention (Accuracy: {accuracy_cross_attn:.4f})')
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Valor Real')

plt.tight_layout()
plt.savefig('mgaf_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Guardar resultados
np.save('mgaf_cross_attention_features_train.npy', X_train_cross_attn)
np.save('mgaf_cross_attention_features_val.npy', X_val_cross_attn)

import pickle
with open('mgaf_cross_attention_best_svm.pkl', 'wb') as f:
    pickle.dump(best_svm_cross_attn, f)

with open('mgaf_traditional_best_svm.pkl', 'wb') as f:
    pickle.dump(best_svm_gaf, f)

print("\nImplementación de MGAF con atención cruzada completada con éxito.")
print(f"Archivos guardados:")
print(f"  - mgaf_cross_attention_features_train.npy")
print(f"  - mgaf_cross_attention_features_val.npy") 
print(f"  - mgaf_cross_attention_best_svm.pkl")
print(f"  - mgaf_traditional_best_svm.pkl")
print(f"  - mgaf_comparison_confusion_matrices.png")