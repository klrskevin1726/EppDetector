import xml.etree.ElementTree as ET
import os
import shutil
import random

# ── Configuración ──────────────────────────────────────────
ANNOTATIONS_DIR = "data/annotations"
IMAGES_DIR      = "data/images"
OUTPUT_DIR      = "data/dataset"
TRAIN_RATIO     = 0.8   # 80% para entrenamiento

# Las clases deben estar en el mismo orden que las encontramos
CLASSES = ["head", "helmet", "person"]

# ── Crear carpetas de salida ────────────────────────────────
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# ── Función que convierte un XML a formato YOLO ─────────────
def convert_xml_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Tamaño de la imagen (necesario para normalizar coordenadas)
    img_width  = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Ignorar clases que no nos interesan
        if class_name not in CLASSES:
            continue

        class_id = CLASSES.index(class_name)

        # Coordenadas del bounding box en píxeles
        xmin = float(obj.find("bndbox/xmin").text)
        ymin = float(obj.find("bndbox/ymin").text)
        xmax = float(obj.find("bndbox/xmax").text)
        ymax = float(obj.find("bndbox/ymax").text)

        # Convertir a formato YOLO (centro + ancho/alto, normalizados)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width    = (xmax - xmin) / img_width
        height   = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines

# ── Procesar todos los archivos ─────────────────────────────
xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
random.seed(42)   # Para que la división sea reproducible
random.shuffle(xml_files)

split_index = int(len(xml_files) * TRAIN_RATIO)
train_files = xml_files[:split_index]
val_files   = xml_files[split_index:]

def process_files(file_list, split):
    converted = 0
    skipped   = 0

    for xml_file in file_list:
        base_name = os.path.splitext(xml_file)[0]
        xml_path  = os.path.join(ANNOTATIONS_DIR, xml_file)

        # Buscar la imagen correspondiente (.jpg o .png)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(IMAGES_DIR, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        # Convertir anotaciones
        yolo_lines = convert_xml_to_yolo(xml_path)
        if not yolo_lines:
            skipped += 1
            continue

        # Copiar imagen a la carpeta correspondiente
        img_ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", split, base_name + img_ext))

        # Guardar el .txt con las anotaciones YOLO
        label_path = os.path.join(OUTPUT_DIR, "labels", split, base_name + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"  {split}: {converted} imágenes convertidas, {skipped} omitidas")

print("Convirtiendo dataset...")
process_files(train_files, "train")
process_files(val_files,   "val")
print("\n✅ Conversión completada!")
print(f"Estructura guardada en: {OUTPUT_DIR}")