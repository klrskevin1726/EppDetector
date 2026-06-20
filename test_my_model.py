from ultralytics import YOLO

# Cargar TU modelo entrenado, no el genérico
model = YOLO("best.pt")

# Correr inferencia sobre tu imagen de prueba
results = model("data/images/test_image.jpg", conf=0.5)  # conf=0.5 filtra detecciones poco confiables

# Mostrar la imagen con los bounding boxes
results[0].show()

# Guardar la imagen anotada en disco
results[0].save("data/images/test_image_result.jpg")

# Imprimir el detalle de cada detección
print("\n--- Detecciones de tu modelo EPP ---")
for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    confidence = float(box.conf)
    print(f"  {class_name}: {confidence:.2%}")