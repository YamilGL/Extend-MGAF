import os
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import torchvision.transforms as transforms

# Rutas
input_dir = 'dataset/Inertial'
output_dir = 'dataset/InertialImages_64x64'

# Crear carpeta de salida principal
os.makedirs(output_dir, exist_ok=True)

# Posibles claves en el archivo
POSSIBLE_KEYS = ['d_iner']

# Definición de configuración según el paper MGAF
# Las señales inerciales tienen 6 componentes: 3 del acelerómetro (Ax, Ay, Az) y 3 del giroscopio (Gx, Gy, Gz)
# El paper describe un método específico para crear imágenes a partir de estas señales

# Orden de combinación según el paper (todas las posibles combinaciones de señales)
# Para las 6 señales (1-indexadas como en el paper), las combinaciones son:
signal_combinations = [
    [0, 1, 2, 3, 4, 5],  # Todas las señales en orden original
    [0, 2, 4, 1, 3, 5],  # Agrupación alternada (acelerómetro, giroscopio)
    [0, 3, 1, 4, 2, 5],  # Combinación cruzada
    [0, 4, 1, 5, 0, 5]   # Últimas filas con repetición de algunas señales
]

# Crear matriz de orden de filas (24 filas según paper)
row_order = []
for combo in signal_combinations:
    row_order.extend(combo)

# Lista de archivos .mat
mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

def extract_inertial_data(mat_data):
    """Extrae los datos inerciales del archivo .mat"""
    for key in POSSIBLE_KEYS:
        if key in mat_data:
            return mat_data[key]
    raise KeyError(f"No se encontró ninguna clave válida: {POSSIBLE_KEYS}")

def get_class_from_filename(filename):
    """Extrae la clase desde el nombre del archivo (por ejemplo: a1_s1_t1.mat -> clase = a1)"""
    return filename.split('_')[0]  # "a1_s1_t1.mat" -> "a1"

def preprocess_inertial_signal(signal_data):
    """
    Preprocesa las señales inerciales según el método descrito en el paper MGAF:
    1. Filtrado para reducir ruido
    2. Normalización
    3. Construcción de la imagen de señal combinada
    """
    # Paso 1: Filtrado para reducir ruido (filtro de mediana como menciona el paper)
    filtered_signals = np.zeros_like(signal_data)
    for i in range(signal_data.shape[0]):
        filtered_signals[i] = medfilt(signal_data[i], kernel_size=5)
    
    # Paso 2: Normalización por señal
    normalized_signals = np.zeros_like(filtered_signals, dtype=np.float32)
    for i in range(filtered_signals.shape[0]):
        signal = filtered_signals[i]
        if np.max(signal) - np.min(signal) > 0:  # Evitar división por cero
            normalized_signals[i] = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    return normalized_signals

def create_signal_image(signal_data, desired_length=52):
    """
    Crea una imagen de señal según el método del paper MGAF:
    1. Ajusta la longitud temporal (truncando o rellenando)
    2. Crea una imagen combinando señales según el patrón especificado
    3. Redimensiona a 64x64
    """
    # Asegurar que tenemos 6 señales
    if signal_data.shape[0] != 6:
        raise ValueError(f"Se esperaban 6 señales, pero se encontraron {signal_data.shape[0]}")
    
    # Ajustar longitud temporal a desired_length (52 como en tu código, que es consistente con el paper)
    if signal_data.shape[1] < desired_length:
        # Rellenar con ceros si es más corto
        pad_width = desired_length - signal_data.shape[1]
        signal_data = np.pad(signal_data, ((0, 0), (0, pad_width)), mode='constant')
    elif signal_data.shape[1] > desired_length:
        # Truncar si es más largo
        signal_data = signal_data[:, :desired_length]
    
    # Crear imagen de señal 24x52 según el patrón de combinación
    signal_image = np.zeros((24, desired_length))
    for i, row_idx in enumerate(row_order):
        signal_image[i] = signal_data[row_idx]
    
    # Normalizar toda la imagen a [0, 255] para visualización
    if np.max(signal_image) - np.min(signal_image) > 0:
        signal_image = 255 * (signal_image - np.min(signal_image)) / (np.max(signal_image) - np.min(signal_image))
    
    # Convertir a uint8
    signal_image = np.uint8(signal_image)
    
    # Redimensionar a 64x64 como se menciona en el paper
    signal_image_resized = cv2.resize(signal_image, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    return signal_image_resized

# Transformaciones para aumento de datos como describe el paper
# El paper menciona que utilizan flip horizontal, rotaciones pequeñas y fluctuaciones menores
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    # No se usa ColorJitter ya que podría afectar la relación entre señales
])

print("Iniciando procesamiento de datos inerciales según paper MGAF...")

# Procesar cada archivo
for mat_file in tqdm(mat_files, desc="Procesando datos inerciales"):
    mat_path = os.path.join(input_dir, mat_file)
    
    try:
        mat_data = scipy.io.loadmat(mat_path)
        inertial_data = extract_inertial_data(mat_data)
        
        # Asegurar que los datos tienen la forma correcta (6 señales x tiempo)
        # El paper menciona que usan 3 señales de acelerómetro y 3 de giroscopio
        inertial_data = inertial_data.T  # Transponer para tener señales como filas
        
        if inertial_data.shape[0] != 6:
            print(f"[SKIPPED] {mat_file}: Se esperaban 6 señales, pero se encontró {inertial_data.shape}")
            continue
            
    except Exception as e:
        print(f"[ERROR] No se pudo procesar {mat_file}: {e}")
        continue
    
    # Paso 1: Preprocesar señales
    processed_signals = preprocess_inertial_signal(inertial_data)
    
    # Paso 2: Crear imagen de señal
    try:
        signal_image = create_signal_image(processed_signals)
    except Exception as e:
        print(f"[ERROR] No se pudo crear la imagen para {mat_file}: {e}")
        continue
    
    # Obtener información de clase y muestra
    sample_name = os.path.splitext(mat_file)[0]
    class_name = get_class_from_filename(sample_name)
    
    # Crear carpeta para la clase
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Guardar imagen original
    base_path = os.path.join(class_output_dir, f'{sample_name}.png')
    cv2.imwrite(base_path, signal_image)
    
    # Generar y guardar imágenes aumentadas
    pil_image = Image.fromarray(signal_image)
    
    # Según el paper MGAF, también aplican técnicas de aumento de datos
    for i in range(1, 4):  # 3 versiones aumentadas como en tu código original
        try:
            augmented = augmentation(pil_image)
            aug_path = os.path.join(class_output_dir, f'{sample_name}_aug{i}.png')
            augmented.save(aug_path)
        except Exception as e:
            print(f"[WARNING] Error en aumento de datos para {mat_file}: {e}")
    
    # Además, según el paper, también generan representaciones adicionales:
    
    # 1. Representación de magnitud - combina señales por magnitud
    try:
        # Combinar señales de acelerómetro por magnitud
        accel_magnitude = np.sqrt(np.sum(processed_signals[:3]**2, axis=0))
        # Combinar señales de giroscopio por magnitud
        gyro_magnitude = np.sqrt(np.sum(processed_signals[3:]**2, axis=0))
        
        # Crear una imagen 2x52 con las magnitudes
        magnitude_image = np.vstack([accel_magnitude, gyro_magnitude])
        
        # Normalizar y redimensionar
        if np.max(magnitude_image) - np.min(magnitude_image) > 0:
            magnitude_image = 255 * (magnitude_image - np.min(magnitude_image)) / (np.max(magnitude_image) - np.min(magnitude_image))
        magnitude_image = np.uint8(magnitude_image)
        magnitude_image = cv2.resize(magnitude_image, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # Guardar
        magnitude_path = os.path.join(class_output_dir, f'{sample_name}_magnitude.png')
        cv2.imwrite(magnitude_path, magnitude_image)
    except Exception as e:
        print(f"[WARNING] Error al crear imagen de magnitud para {mat_file}: {e}")
    
    # 2. Representación de correlación - correlación entre señales
    try:
        # Calcular matriz de correlación entre las 6 señales
        corr_matrix = np.corrcoef(processed_signals)
        
        # Normalizar y convertir a imagen
        corr_image = 255 * (corr_matrix - np.min(corr_matrix)) / (np.max(corr_matrix) - np.min(corr_matrix))
        corr_image = np.uint8(corr_image)
        corr_image = cv2.resize(corr_image, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # Guardar
        corr_path = os.path.join(class_output_dir, f'{sample_name}_correlation.png')
        cv2.imwrite(corr_path, corr_image)
    except Exception as e:
        print(f"[WARNING] Error al crear imagen de correlación para {mat_file}: {e}")

print("✅ Procesamiento de datos inerciales completado según metodología MGAF del paper.")
print("Se han generado imágenes de señal, magnitud y correlación, más aumentos de datos.")