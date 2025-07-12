# ==============================================================================
# SCRIPT UNIFICADO DE PROCESAMIENTO Y DIVISIÓN DE DATASET DE PROFUNDIDAD
# ==============================================================================
#
# Este script realiza un pipeline completo:
# 1.  Lee archivos de datos de profundidad en formato .mat.
# 2.  Convierte cada secuencia de profundidad en imágenes:
#     -   Sequential Frame Images (SFI): una imagen por cada frame de la secuencia.
#     -   Motion Energy Image (MEI): una imagen que resume el movimiento total.
#     -   Motion History Image (MHI): una imagen que visualiza la cronología del movimiento.
# 3.  Guarda estas imágenes en un directorio temporal.
# 4.  Divide el conjunto de imágenes generado en subconjuntos de `train`, `validation`,
#     y `test`, asegurando que todos los frames de una misma muestra original
#     (ej. 'a1_s1_v1') permanezcan en el mismo subconjunto.
# 5.  (Opcional) Limpia el directorio temporal después de la división.
#
# Uso:
#   -   Ajusta los parámetros en la sección "Parámetros de Configuración".
#   -   Asegúrate de tener los datos de profundidad en `DEPTH_DATA_DIR`.
#   -   Ejecuta el script: `python tu_nombre_de_script.py`
#
# ==============================================================================

import os
import shutil
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- Parámetros de Configuración ---

# -- Parte 1: Generación de Imágenes --
# Directorio con los datos de profundidad originales (.mat)
DEPTH_DATA_DIR = 'datasets/UTD-MHAD/Depth'
# Directorio temporal para almacenar las imágenes generadas antes de dividirlas.
# Este directorio será creado y (opcionalmente) eliminado por el script.
TEMP_PROCESSED_DIR = 'datasets/Processed/Temp_Depth_Images'
# Tamaño al que se redimensionarán las imágenes generadas
IMG_SIZE = (64, 64)
# ¿Generar MEI y MHI además de los frames individuales (SFI)?
GENERATE_MOTION_IMAGES = True

# -- Parte 2: División del Dataset --
# Directorio final donde se guardarán las carpetas train/validation/test
FINAL_OUTPUT_DIR = 'datasets/Split_Data_Depth_64x64'
# Proporciones de la división (deben sumar 1.0)
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
# Semilla aleatoria para que la división sea reproducible
RANDOM_SEED = 42

# -- Parte 3: Limpieza --
# ¿Eliminar el directorio temporal (TEMP_PROCESSED_DIR) al finalizar?
CLEANUP_TEMP_DIR = True


# --- Funciones Auxiliares ---

def get_info_from_mat_filename(filename):
    """Extrae clase y nombre de la muestra del nombre de archivo .mat."""
    parts = os.path.splitext(filename)[0].split('_')
    class_name = parts[0]
    sample_name = '_'.join(parts[:-1]) # e.g., a1_s1_v1
    return class_name, sample_name

def get_sample_name_from_image_filename(filename):
    """Extrae el nombre de la muestra base de un archivo de imagen generado."""
    # Asume el formato aX_sY_vZ
    return '_'.join(filename.split('_')[:3])

def preprocess_depth_frame(frame):
    """Normaliza y redimensiona un único frame de profundidad."""
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_min, frame_max = np.min(frame), np.max(frame)
    if frame_max - frame_min > 1e-6:
        norm_frame = 255 * (frame - frame_min) / (frame_max - frame_min)
    else:
        norm_frame = np.zeros_like(frame)
    frame_uint8 = norm_frame.astype(np.uint8)
    resized_frame = cv2.resize(frame_uint8, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized_frame

def generate_motion_images(sfi_sequence):
    """Genera MEI y MHI a partir de una secuencia de SFIs."""
    if len(sfi_sequence) < 2:
        return {}

    sfi_array = np.array(sfi_sequence, dtype=np.float32)
    
    # Motion Energy Image (MEI)
    mei = np.max(np.abs(np.diff(sfi_array, axis=0)), axis=0)
    if mei.max() > 0:
        mei = (255 * (mei / mei.max())).astype(np.uint8)
    else:
        mei = np.zeros_like(sfi_sequence[0])

    # Motion History Image (MHI)
    mhi = np.zeros_like(sfi_sequence[0], dtype=np.float32)
    tau = len(sfi_sequence)  # Duración de la "memoria"
    for i, frame in enumerate(sfi_sequence):
        # Umbral adaptativo para detectar movimiento
        motion_threshold = np.percentile(frame, 85)
        motion_mask = frame > motion_threshold
        # Actualizar MHI
        mhi = mhi - 1.0 / tau  # Decaer la historia
        mhi[mhi < 0] = 0
        mhi[motion_mask] = 1.0 # Reiniciar la historia donde hay movimiento
        
    mhi_normalized = (255 * mhi).astype(np.uint8)

    return {
        'mei': Image.fromarray(mei, 'L'),
        'mhi': Image.fromarray(mhi_normalized, 'L')
    }

# --- Funciones del Pipeline ---

def step1_generate_images():
    """Paso 1: Convierte datos de profundidad .mat a imágenes."""
    print("🚀 PASO 1: Iniciando la conversión de datos de profundidad a imágenes...")
    if not os.path.exists(DEPTH_DATA_DIR):
        print(f"❌ ERROR: El directorio de entrada '{DEPTH_DATA_DIR}' no existe.")
        return False, 0

    os.makedirs(TEMP_PROCESSED_DIR, exist_ok=True)
    depth_files = [f for f in os.listdir(DEPTH_DATA_DIR) if f.endswith('.mat')]
    
    if not depth_files:
        print(f"❌ ERROR: No se encontraron archivos .mat en '{DEPTH_DATA_DIR}'.")
        return False, 0
        
    print(f"📁 Encontrados {len(depth_files)} archivos de profundidad. Procesando...")
    archivos_creados_count = 0

    for filename in tqdm(depth_files, desc="Generando Imágenes"):
        try:
            class_name, sample_name = get_info_from_mat_filename(filename)
            class_dir = os.path.join(TEMP_PROCESSED_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            mat_path = os.path.join(DEPTH_DATA_DIR, filename)
            depth_sequence = scipy.io.loadmat(mat_path)['d_depth']
            
            num_frames = depth_sequence.shape[2]
            sfi_list = []

            for i in range(num_frames):
                frame = depth_sequence[:, :, i]
                sfi_frame = preprocess_depth_frame(frame)
                sfi_list.append(sfi_frame)
                img = Image.fromarray(sfi_frame, 'L')
                output_path = os.path.join(class_dir, f"{sample_name}_sfi_frame{i:03d}.png")
                img.save(output_path)
                archivos_creados_count += 1

            if GENERATE_MOTION_IMAGES and sfi_list:
                motion_imgs = generate_motion_images(sfi_list)
                for name, img in motion_imgs.items():
                    output_path = os.path.join(class_dir, f"{sample_name}_{name}.png")
                    img.save(output_path)
                    archivos_creados_count += 1

        except Exception as e:
            print(f"\n⚠️  Error procesando {filename}: {e}")

    print("\n✅ Paso 1 completado.")
    print(f"📊 Total de imágenes generadas: {archivos_creados_count}")
    print(f"🖼️  Imágenes guardadas temporalmente en: '{TEMP_PROCESSED_DIR}'")
    return True, archivos_creados_count

def step2_split_dataset():
    """Paso 2: Divide las imágenes generadas en train/validation/test."""
    print("\n🚀 PASO 2: Iniciando la división del dataset...")

    # 1. Crear directorios de salida finales
    train_path = os.path.join(FINAL_OUTPUT_DIR, 'train')
    val_path = os.path.join(FINAL_OUTPUT_DIR, 'validation')
    test_path = os.path.join(FINAL_OUTPUT_DIR, 'test')
    shutil.rmtree(FINAL_OUTPUT_DIR, ignore_errors=True) # Limpiar antes de empezar
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    print(f"📁 Directorios de salida finales creados en: '{FINAL_OUTPUT_DIR}'")

    # 2. Agrupar archivos por muestra para cada clase
    class_dirs = [d for d in os.listdir(TEMP_PROCESSED_DIR) if os.path.isdir(os.path.join(TEMP_PROCESSED_DIR, d))]
    
    if not class_dirs:
        print(f"❌ ERROR: No se encontraron directorios de clases en '{TEMP_PROCESSED_DIR}'.")
        return False

    total_files_moved = 0
    print(f"🧠 Analizando {len(class_dirs)} clases para la división...")

    for class_name in tqdm(class_dirs, desc="Dividiendo Clases"):
        class_dir_path = os.path.join(TEMP_PROCESSED_DIR, class_name)
        sample_files = defaultdict(list)
        for filename in os.listdir(class_dir_path):
            sample_name = get_sample_name_from_image_filename(filename)
            sample_files[sample_name].append(filename)
        
        unique_samples = list(sample_files.keys())

        if len(unique_samples) < 3:
            print(f"\n⚠️  ADVERTENCIA: La clase '{class_name}' tiene solo {len(unique_samples)} muestras únicas. Se asignarán todas a 'train'.")
            train_samples, val_samples, test_samples = unique_samples, [], []
        else:
            train_samples, temp_samples = train_test_split(
                unique_samples, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
            )
            relative_val_ratio = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO) if (VALIDATION_RATIO + TEST_RATIO) > 0 else 0.0
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=(1 - relative_val_ratio), random_state=RANDOM_SEED
            )
        
        datasets = {
            'train': (train_samples, train_path),
            'validation': (val_samples, val_path),
            'test': (test_samples, test_path)
        }

        for set_name, (samples, dest_base_path) in datasets.items():
            dest_class_path = os.path.join(dest_base_path, class_name)
            os.makedirs(dest_class_path, exist_ok=True)
            for sample_name in samples:
                for filename in sample_files[sample_name]:
                    source_file = os.path.join(class_dir_path, filename)
                    dest_file = os.path.join(dest_class_path, filename)
                    shutil.move(source_file, dest_file)
                    total_files_moved += 1

    print("\n✅ Paso 2 completado.")
    print(f"📊 Total de archivos movidos: {total_files_moved}")
    print(f"📂 Los conjuntos de datos están listos en:")
    print(f"   - Entrenamiento: {train_path}")
    print(f"   - Validación:    {val_path}")
    print(f"   - Prueba:        {test_path}")
    return True

def step3_cleanup():
    """Paso 3: Limpia el directorio temporal."""
    if CLEANUP_TEMP_DIR:
        print(f"\n🚀 PASO 3: Limpiando el directorio temporal...")
        try:
            shutil.rmtree(TEMP_PROCESSED_DIR)
            print(f"✅ Directorio temporal '{TEMP_PROCESSED_DIR}' eliminado con éxito.")
        except OSError as e:
            print(f"❌ Error al eliminar el directorio temporal: {e}")
    else:
        print("\nℹ️  Paso 3 omitido: La limpieza del directorio temporal está desactivada.")


# --- Script Principal ---

def main():
    """Ejecuta el pipeline completo de procesamiento y división."""
    
    # Validar que los ratios sumen 1.0 para evitar errores
    if not abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9:
        print("❌ ERROR: La suma de TRAIN_RATIO, VALIDATION_RATIO y TEST_RATIO debe ser 1.0.")
        return

    # Ejecutar Paso 1
    success_step1, _ = step1_generate_images()
    if not success_step1:
        print("\nEl pipeline se detuvo debido a un error en el Paso 1.")
        return

    # Ejecutar Paso 2
    success_step2 = step2_split_dataset()
    if not success_step2:
        print("\nEl pipeline se detuvo debido a un error en el Paso 2.")
        return
        
    # Ejecutar Paso 3
    step3_cleanup()

    print("\n🎉 ¡PIPELINE COMPLETADO CON ÉXITO! 🎉")

if __name__ == '__main__':
    main()