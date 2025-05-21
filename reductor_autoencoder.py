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
#   MGAF IMPLEMENTACIÓN
# ---------------------------- #

def gated_average_fusion(F1, F2):
    # Ajustar dimensiones para que coincidan
    if F1.shape[0] != F2.shape[0]:
        # Elegir la dimensión más pequeña
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
    
    Args:
        model: Modelo CNN del que extraer características
        dataset: Conjunto de datos para extraer características
        layer_names: Lista de nombres de capas de las cuales extraer características
        device: Dispositivo en el que ejecutar el modelo
        
    Returns:
        Dictionary de características por capa y etiquetas
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
        # Entrada: 64x64
        # Conv1: 60x60 -> Pool1: 30x30
        # Conv2: 26x26
        # Conv3: 22x22 -> Pool2: 11x11
        flat_size = 32 * 11 * 11
        
        # Capa fully connected (FC1)
        self.fc1 = nn.Linear(flat_size, 128)
        
        # Capa de salida
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Flujo forward pasando por todas las capas con activaciones ReLU
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
    transforms.Grayscale(),  # Imágenes a escala de grises
    transforms.Resize((64, 64)),  # Redimensionar a 64x64 como en el paper
    transforms.ToTensor()
])

# Cargar datasets
depth_dataset = datasets.ImageFolder(root='dataset/DepthImages_64x64', transform=transform)
inertial_dataset = datasets.ImageFolder(root='dataset/InertialImages_64x64', transform=transform)

# Balanceo del dataset de profundidad (opcional según el paper)
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

# Número de clases según el dataset
NUM_CLASSES = 27  # Según el paper o ajustar a tu dataset específico

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

# Capas para extraer características según el paper MGAF
layers = ['conv1', 'conv2', 'conv3', 'fc1']

# Extracción de características del modelo de profundidad
print("Extrayendo características del modelo de profundidad...")
depth_features_train, y_train_depth = extract_features(depth_model, depth_train, layers, device)
depth_features_val, y_val_depth = extract_features(depth_model, depth_val, layers, device)

# Extracción de características del modelo inercial
print("Extrayendo características del modelo inercial...")
inertial_features_train, y_train_inertial = extract_features(inertial_model, inertial_train, layers, device)
inertial_features_val, y_val_inertial = extract_features(inertial_model, inertial_val, layers, device)

print("Fusionando características con MGAF...")

# ---------------------------- #
#   MULTISTAGE GATED AVERAGE FUSION (MGAF)
# ---------------------------- #

# Fusión por capa y concatenación para entrenamiento
fused_train_features = []
for layer in layers:
    fused_layer = gated_average_fusion(
        depth_features_train[layer], 
        inertial_features_train[layer]
    )
    fused_train_features.append(fused_layer)

# Fusión por capa y concatenación para validación
fused_val_features = []
for layer in layers:
    fused_layer = gated_average_fusion(
        depth_features_val[layer], 
        inertial_features_val[layer]
    )
    fused_val_features.append(fused_layer)

# Concatenar todas las características fusionadas
X_train = np.concatenate(fused_train_features, axis=1)
X_val = np.concatenate(fused_val_features, axis=1)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder(X_train.shape[1], 64).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
for epoch in range(100):
    autoencoder.train()
    encoded, decoded = autoencoder(X_tensor)
    loss = criterion(decoded, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


autoencoder.eval()
with torch.no_grad():
    X_train_reduced = autoencoder.encoder(X_tensor).cpu().numpy()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_val_reduced = autoencoder.encoder(X_val_tensor).cpu().numpy()


X_train =X_train_reduced
X_val = X_val_reduced

y_train = y_train_inertial  # Usamos las etiquetas de cualquier modalidad, ya que deben coincidir
y_val = y_val_inertial

print(f"Dimensión de características fusionadas - Train: {X_train.shape}, Val: {X_val.shape}")

# ---------------------------- #
#   CLASIFICACIÓN CON SVM
# ---------------------------- #

print("Entrenando SVM con búsqueda de hiperparámetros...")

# Búsqueda de hiperparámetros para SVM como se menciona en el paper
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(
    SVC(), 
    param_grid, 
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_

print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

# Evaluar en conjunto de validación
y_pred = best_svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Precisión en conjunto de validación: {accuracy:.4f}")

# ---------------------------- #
#   MATRIZ DE CONFUSIÓN
# ---------------------------- #

conf_mat = confusion_matrix(y_val, y_pred)
norm_conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(norm_conf_mat, annot=True, fmt=".2f", cmap='Blues')
plt.title(f'Matriz de Confusión Normalizada - MGAF (Accuracy: {accuracy:.4f})')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('mgaf_confusion_matrix.png')
plt.show()

# Guardar resultados y modelo
np.save('mgaf_features_train.npy', X_train)
np.save('mgaf_features_val.npy', X_val)
np.save('mgaf_labels_train.npy', y_train)
np.save('mgaf_labels_val.npy', y_val)

import pickle
with open('mgaf_best_svm.pkl', 'wb') as f:
    pickle.dump(best_svm, f)

print("Implementación de MGAF completada con éxito.")