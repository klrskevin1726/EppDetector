import gradio as gr
from ultralytics import YOLO

# Cargar el modelo una sola vez al iniciar la app
model = YOLO("best.pt")

def detect_epp(image, confidence):
    """
    Receives an image and a confidence threshold, runs the EPP detection
    model, and returns the annotated image plus a summary of detections.
    """
    if image is None:
        return None, "Please upload an image first."

    # Run inference with the user-selected confidence threshold
    results = model(image, conf=confidence)

    # Draw bounding boxes on the image (already in RGB order)
    annotated_image = results[0].plot()

    # Count detections per class
    boxes = results[0].boxes
    if len(boxes) == 0:
        summary = "⚠️ No PPE detected in this image. Try lowering the confidence threshold."
    else:
        counts = {}
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

        # Build a readable summary like "2 helmet(s), 1 head"
        summary_parts = [f"{count} {name}(s)" for name, count in counts.items()]
        summary = "✅ Detected: " + ", ".join(summary_parts)

    return annotated_image, summary

# Build the interface
demo = gr.Interface(
    fn=detect_epp,
    inputs=[
        gr.Image(type="numpy", label="Upload an image"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Confidence threshold"
        )
    ],
    outputs=[
        gr.Image(type="numpy", label="PPE Detection"),
        gr.Textbox(label="Summary")
    ],
    title="🦺 PPE (Personal Protective Equipment) Detector",
    description=(
        "Upload a photo of workers to detect whether they are wearing a helmet. "
        "YOLOv8 model trained on the Safety Helmet Detection dataset (Kaggle)."
    ),
    examples=[["data/images/test_image.jpg", 0.5]],
    article=(
        "**Note on limitations:** this model was trained on images where helmets "
        "are worn on the head. It may produce false positives when a helmet is "
        "held in someone's hand rather than worn, since that pattern was not "
        "well represented in the training dataset."
    )
)

if __name__ == "__main__":
    demo.launch()