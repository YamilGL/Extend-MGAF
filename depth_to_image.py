import os
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import torchvision.transforms as transforms

# Rutas
input_dir = 'dataset/Depth'
output_dir = 'dataset/DepthImages_64x64'

# Crear carpeta de salida principal
os.makedirs(output_dir, exist_ok=True)

# Lista de archivos .mat en la carpeta
mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

# Posibles claves dentro del archivo .mat
POSSIBLE_KEYS = ['d_depth']

# Transformaciones de aumento como se menciona en el paper
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Rotación, traslación y escala
])

def extract_depth_sequence(mat_data):
    """Extrae la secuencia de profundidad del archivo .mat"""
    for key in POSSIBLE_KEYS:
        if key in mat_data:
            return mat_data[key]
    raise KeyError(f"Ninguna de las claves esperadas {POSSIBLE_KEYS} fue encontrada.")

def get_class_from_filename(filename):
    """Extrae la clase desde el nombre del archivo (por ejemplo: a1_s1_t1.mat -> clase = a1)"""
    return filename.split('_')[0]  # "a1_s1_t1.mat" -> "a1"

def preprocess_depth_frame(depth_frame):
    """
    Preprocesa un frame de profundidad según el método descrito en el paper MGAF:
    1. Aplica filtro de mediana para reducir ruido
    2. Normaliza valores a rango [0,255]
    3. Redimensiona a 64x64
    """
    # Paso 1: Filtro de mediana para reducir ruido (como menciona el paper)
    filtered_frame = medfilt2d(depth_frame, kernel_size=5)
    
    # Paso 2: Normalización
    # El paper menciona que normalizan los valores de profundidad a [0,255]
    if np.max(filtered_frame) - np.min(filtered_frame) > 0:  # Evitar división por cero
        norm_frame = 255 * (filtered_frame - np.min(filtered_frame)) / (np.max(filtered_frame) - np.min(filtered_frame))
    else:
        norm_frame = np.zeros_like(filtered_frame)
    
    # Convertir a uint8 para imagen
    frame_uint8 = np.uint8(norm_frame)
    
    # Paso 3: Redimensionar a 64x64 como menciona el paper
    frame_resized = cv2.resize(frame_uint8, (64, 64), interpolation=cv2.INTER_AREA)
    
    return frame_resized

def generate_mhi(depth_sequence, num_frames=None):
    """
    Genera una Motion History Image (MHI) como se describe en el paper.
    MHI representa la información temporal del movimiento.
    """
    if num_frames is None:
        num_frames = depth_sequence.shape[2]
    else:
        num_frames = min(num_frames, depth_sequence.shape[2])
    
    # Inicializar MHI con ceros
    height, width = depth_sequence.shape[0], depth_sequence.shape[1]
    mhi = np.zeros((height, width), dtype=np.float32)
    
    # Calcular diferencias entre frames consecutivos
    alpha = 1.0 / num_frames  # Factor de decaimiento
    
    for i in range(1, num_frames):
        # Diferencia absoluta entre frames consecutivos
        frame_diff = np.abs(depth_sequence[:,:,i] - depth_sequence[:,:,i-1])
        
        # Actualizar MHI: decaer valores existentes y añadir nuevos movimientos
        mhi = np.maximum(mhi * (1.0 - alpha), frame_diff)
    
    # Normalizar MHI a [0,255]
    if np.max(mhi) > 0:
        mhi = 255 * mhi / np.max(mhi)
    
    return np.uint8(mhi)

print("Iniciando procesamiento de datos de profundidad según paper MGAF...")

# Iterar sobre cada archivo .mat
for mat_file in tqdm(mat_files, desc="Procesando secuencias de profundidad"):
    mat_path = os.path.join(input_dir, mat_file)
    
    try:
        mat_data = scipy.io.loadmat(mat_path)
        depth_sequence = extract_depth_sequence(mat_data)
    except Exception as e:
        print(f"[ERROR] No se pudo procesar {mat_file}: {e}")
        continue

    if depth_sequence.ndim != 3:
        print(f"[SKIPPED] {mat_file}: Dimensión inesperada {depth_sequence.shape}")
        continue

    # Obtener clase y nombre de muestra
    sample_name = os.path.splitext(mat_file)[0]
    class_name = get_class_from_filename(sample_name)
    
    # Crear subcarpeta para la clase
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Según el paper MGAF, utilizan tres representaciones principales:
    
    # 1. DMI (Depth Motion Image): Suma de diferencias absolutas entre frames consecutivos
    dmi = np.zeros_like(depth_sequence[:,:,0], dtype=np.float32)
    for i in range(1, depth_sequence.shape[2]):
        dmi += np.abs(depth_sequence[:,:,i] - depth_sequence[:,:,i-1])
    
    # Normalizar y procesar DMI
    dmi_processed = preprocess_depth_frame(dmi)
    dmi_img = Image.fromarray(dmi_processed)
    dmi_img.save(os.path.join(class_output_dir, f'{sample_name}_dmi.png'))
    
    # 2. MHI (Motion History Image): Representa la historia del movimiento
    mhi = generate_mhi(depth_sequence)
    mhi_processed = preprocess_depth_frame(mhi)
    mhi_img = Image.fromarray(mhi_processed)
    mhi_img.save(os.path.join(class_output_dir, f'{sample_name}_mhi.png'))
    
    # 3. Depth Frame Difference (DFD): Diferencia entre primer y último frame
    if depth_sequence.shape[2] > 1:
        first_frame = depth_sequence[:,:,0]
        last_frame = depth_sequence[:,:,-1]
        dfd = np.abs(last_frame - first_frame)
        dfd_processed = preprocess_depth_frame(dfd)
        dfd_img = Image.fromarray(dfd_processed)
        dfd_img.save(os.path.join(class_output_dir, f'{sample_name}_dfd.png'))
    
    # 4. Proyecciones (como menciona el paper): Frame medio procesado
    mean_frame = np.mean(depth_sequence, axis=2)
    mean_processed = preprocess_depth_frame(mean_frame)
    mean_img = Image.fromarray(mean_processed)
    mean_img.save(os.path.join(class_output_dir, f'{sample_name}_mean.png'))
    
    # Aplicar aumentos de datos como se menciona en el paper
    # El paper destaca que usan técnicas de aumento como flip horizontal, rotación, traslación
    pil_image = Image.fromarray(mean_processed)
    
    # Generar versiones aumentadas (siguiendo el paper)
    for i in range(1, 4):  # Generar 3 aumentos diferentes
        aug_image = augmentation(pil_image)
        aug_image.save(os.path.join(class_output_dir, f'{sample_name}_aug{i}.png'))

print("✅ Procesamiento completo según metodología MGAF del paper.")
print("Se han generado múltiples representaciones por cada secuencia: DMI, MHI, DFD y proyección media.")
print("También se aplicaron aumentos de datos consistentes con el paper.")