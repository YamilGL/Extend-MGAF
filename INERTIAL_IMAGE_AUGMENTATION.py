import os
import scipy.io
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from scipy.signal import medfilt
from scipy.ndimage import prewitt

# ==== Configuración ====
input_dir = 'dataset/Inertial'
output_dir = 'dataset/InertialImages_64x64_Extended'
os.makedirs(output_dir, exist_ok=True)
mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
POSSIBLE_KEYS = ['d_iner']

# ==== Aumentos ====
augmentation = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    #transforms.RandomRotation(degrees=15),
])

# ==== Funciones ====
def extract_inertial_data(mat_data):
    for key in POSSIBLE_KEYS:
        if key in mat_data:
            return mat_data[key]
    raise KeyError(f"Clave no encontrada. Intentado con {POSSIBLE_KEYS}.")

def get_class_from_filename(filename):
    return filename.split('_')[0]  # a1_s1_t1.mat -> a1

def preprocess_inertial_segment(segment_data):
    """
    Preprocesa un segmento de señales inerciales:
    - Filtrado
    - Normalización
    - Conversión a imagen 64x64
    """
    # Filtrado para reducir ruido
    filtered_signals = np.zeros_like(segment_data, dtype=np.float32)
    for i in range(segment_data.shape[0]):
        filtered_signals[i] = medfilt(segment_data[i], kernel_size=5)
    
    # Normalización por señal
    for i in range(filtered_signals.shape[0]):
        signal = filtered_signals[i]
        if np.max(signal) - np.min(signal) > 0:
            filtered_signals[i] = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    # Convertir a imagen: 6 señales x tiempo -> imagen cuadrada
    # Si tenemos menos tiempo del necesario, hacer padding
    target_size = max(64, int(np.sqrt(segment_data.shape[0] * segment_data.shape[1])))
    
    # Crear imagen concatenando señales
    if segment_data.shape[1] < target_size:
        # Padding si es necesario
        pad_width = target_size - segment_data.shape[1]
        padded = np.pad(filtered_signals, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncar si es muy largo
        padded = filtered_signals[:, :target_size]
    
    # Crear imagen apilando las señales verticalmente
    signal_image = padded
    
    # Normalizar a [0, 255]
    if np.max(signal_image) - np.min(signal_image) > 0:
        signal_image = 255 * (signal_image - np.min(signal_image)) / (np.max(signal_image) - np.min(signal_image))
    
    signal_image = np.uint8(signal_image)
    
    # Redimensionar a 64x64
    resized = cv2.resize(signal_image, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    return resized

def apply_filters(signal_image):
    """Aplica filtros de procesamiento de señales a la imagen"""
    
    # Sobel (detección de cambios bruscos en señales)
    #sobelx = cv2.Sobel(signal_image, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(signal_image, cv2.CV_64F, 0, 1, ksize=5)
    #sobel = np.sqrt(sobelx**2 + sobely**2)
    #sobel_norm = np.uint8(255 * sobel / np.max(sobel)) if np.max(sobel) != 0 else np.zeros_like(signal_image)

    # Prewitt (otro detector de cambios)
    prew = np.sqrt(prewitt(signal_image, axis=0)**2 + prewitt(signal_image, axis=1)**2)
    prew_norm = np.uint8(255 * prew / np.max(prew)) if np.max(prew) != 0 else np.zeros_like(signal_image)

    # CLAHE (mejora de contraste para destacar patrones)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #clahe_img = clahe.apply(signal_image)

    return {'prewitt': prew_norm}
    #return {'sobel': sobel_norm, 'prewitt': prew_norm, 'clahe': clahe_img}

# ==== Proceso principal ====
print("🔄 Generando imágenes de señales inerciales con procedimiento extendido...")
image_count = 0  # Contador de imágenes

for mat_file in tqdm(mat_files, desc="Procesando secuencias inerciales"):
    try:
        mat_path = os.path.join(input_dir, mat_file)
        mat_data = scipy.io.loadmat(mat_path)
        inertial_data = extract_inertial_data(mat_data)
        
        # Transponer para tener señales como filas (6 señales x tiempo)
        inertial_data = inertial_data.T
        
        if inertial_data.shape[0] != 6:
            print(f"[SKIPPED] {mat_file}: Se esperaban 6 señales, se encontraron {inertial_data.shape[0]}")
            continue

        class_name = get_class_from_filename(mat_file)
        sample_name = os.path.splitext(mat_file)[0]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Extraer múltiples segmentos de la secuencia temporal (cada 50 muestras, similar a cada 5 frames)
        segment_size = 50
        for i in range(0, inertial_data.shape[1], segment_size):
            # Extraer segmento
            end_idx = min(i + segment_size, inertial_data.shape[1])
            segment = inertial_data[:, i:end_idx]
            
            # Si el segmento es muy pequeño, saltar
            if segment.shape[1] < 10:
                continue
            
            # Preprocesar segmento
            segment_proc = preprocess_inertial_segment(segment)
            pil_img = Image.fromarray(segment_proc)

            # Base name para este segmento
            base_name = f"{sample_name}_s{i}"

            # 1. Guardar segmento base
            pil_img.save(os.path.join(class_dir, f"{base_name}.png"))
            image_count += 1

            # 2. Aplicar filtros y guardar
            filtered = apply_filters(segment_proc)
            for filter_name, filtered_img in filtered.items():
                Image.fromarray(filtered_img).save(os.path.join(class_dir, f"{base_name}_{filter_name}.png"))
                image_count += 1

                # Aplicar aumentos a cada filtro
                pil_filtered = Image.fromarray(filtered_img)
                for j in range(1, 4):  # 3 versiones aumentadas
                    try:
                        aug_img = augmentation(pil_filtered)
                        aug_img.save(os.path.join(class_dir, f"{base_name}_{filter_name}_aug{j}.png"))
                        image_count += 1
                    except Exception as e:
                        print(f"[WARNING] Error en aumento {filter_name} para {mat_file}: {e}")


    except Exception as e:
        print(f"❌ Error con {mat_file}: {e}")
        continue

print("✅ Finalizado: imágenes de señales inerciales generadas con procedimiento extendido.")
print("Se han generado segmentos base, filtros sobel/prewitt/clahe, más aumentos de datos.")

print(f"✅ Finalizado: se generaron {image_count} imágenes en total.")