import os
import scipy.io
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from scipy.signal import medfilt2d
from scipy.ndimage import prewitt

# ==== Configuración ====
input_dir = 'dataset/Depth'
output_dir = 'dataset/DepthImages_64x64_Extended'
os.makedirs(output_dir, exist_ok=True)
mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
POSSIBLE_KEYS = ['d_depth']

# ==== Aumentos ====
augmentation = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    #transforms.RandomRotation(degrees=15),
])

# ==== Funciones ====
def extract_depth_sequence(mat_data):
    for key in POSSIBLE_KEYS:
        if key in mat_data:
            return mat_data[key]
    raise KeyError(f"Clave no encontrada. Intentado con {POSSIBLE_KEYS}.")

def get_class_from_filename(filename):
    return filename.split('_')[0]  # a1_s1_t1.mat -> a1

def preprocess_depth_frame(depth_frame):
    filtered = medfilt2d(depth_frame, kernel_size=5)
    if np.max(filtered) - np.min(filtered) > 0:
        norm = 255 * (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    else:
        norm = np.zeros_like(filtered)
    resized = cv2.resize(np.uint8(norm), (64, 64), interpolation=cv2.INTER_AREA)
    return resized

def apply_filters(image):
    # Sobel
    #sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    #sobel = np.sqrt(sobelx**2 + sobely**2)
    #sobel_norm = np.uint8(255 * sobel / np.max(sobel)) if np.max(sobel) != 0 else np.zeros_like(image)

    # Prewitt
    prew = np.sqrt(prewitt(image, axis=0)**2 + prewitt(image, axis=1)**2)
    prew_norm = np.uint8(255 * prew / np.max(prew)) if np.max(prew) != 0 else np.zeros_like(image)

    # CLAHE
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #clahe_img = clahe.apply(image)

    return {'prewitt': prew_norm}
    #return {'sobel': sobel_norm, 'prewitt': prew_norm, 'clahe': clahe_img}

# ==== Proceso principal ====
print("🔄 Generando imágenes extendidas...")
image_count = 0  # Contador de imágenes

for mat_file in tqdm(mat_files, desc="Procesando secuencias"):
    try:
        mat_path = os.path.join(input_dir, mat_file)
        mat_data = scipy.io.loadmat(mat_path)
        depth_sequence = extract_depth_sequence(mat_data)
        if depth_sequence.ndim != 3:
            continue

        class_name = get_class_from_filename(mat_file)
        sample_name = os.path.splitext(mat_file)[0]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Extraer múltiples frames de la secuencia (cada 5 frames, por ejemplo)
        for i in range(0, depth_sequence.shape[2], 5):
            frame = depth_sequence[:, :, i]
            frame_proc = preprocess_depth_frame(frame)
            pil_img = Image.fromarray(frame_proc)

            # Guardar frame base
            base_name = f"{sample_name}_f{i}"
            pil_img.save(os.path.join(class_dir, f"{base_name}.png"))
            image_count += 1

            # Aplicar filtros y guardar
            filtered = apply_filters(frame_proc)
            for filter_name, filtered_img in filtered.items():
                Image.fromarray(filtered_img).save(os.path.join(class_dir, f"{base_name}_{filter_name}.png"))
                image_count += 1

                # Aplicar aumentos a cada filtro
                # pil_filtered = Image.fromarray(filtered_img)
                # for j in range(1, 4):
                #     aug_img = augmentation(pil_filtered)
                #     aug_img.save(os.path.join(class_dir, f"{base_name}_{filter_name}_aug{j}.png"))
                #     image_count += 1

    except Exception as e:
        print(f"❌ Error con {mat_file}: {e}")
        continue

print("✅ Finalizado: imágenes extendidas generadas.")
print(f"✅ Finalizado: se generaron {image_count} imágenes en total.")
