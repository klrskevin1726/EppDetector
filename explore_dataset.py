import xml.etree.ElementTree as ET
import os

# Carpeta donde están los archivos XML
annotations_path = "data/annotations"

# Recolectar todas las clases únicas que aparecen en el dataset
all_classes = set()
total_objects = 0

# Leer todos los XML
xml_files = os.listdir(annotations_path)
print(f"Total de archivos XML encontrados: {len(xml_files)}\n")

# Explorar los primeros 20 para ver qué clases hay
for xml_file in xml_files[:20]:
    tree = ET.parse(os.path.join(annotations_path, xml_file))
    root = tree.getroot()
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        all_classes.add(class_name)
        total_objects += 1

print(f"Clases encontradas en el dataset:")
for i, cls in enumerate(sorted(all_classes)):
    print(f"  [{i}] {cls}")

print(f"\nTotal de objetos anotados (muestra de 20 imágenes): {total_objects}")