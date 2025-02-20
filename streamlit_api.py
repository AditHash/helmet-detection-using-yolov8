import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
MODEL_PATH = r"C:\helmet\best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Helmet Detection App")
st.write("Upload an image to detect helmets.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Save to a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run YOLO model on the image
    results = model(temp_file_path)

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            
            # Bounding box color
            color = (0, 255, 0) if cls == 1 else (0, 0, 255)  # Green for "With Helmet", Red for "Without Helmet"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

            # Prepare label text
            label = f"{model.names[cls]}: {round(conf * 100, 1)}%"  # Confidence in percentage

            # Font settings (larger size)
            font_scale = 2  # Increased font size
            font_thickness = 5  # Thicker text
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Ensure label is visible (put background)
            text_bg_x1, text_bg_y1 = x1, y1 - text_height - 15
            text_bg_x2, text_bg_y2 = x1 + text_width + 10, y1

            # Draw filled rectangle as background for text
            cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

            # Put text over the box
            cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    # Convert back to PIL image for display
    image = Image.fromarray(image)

    # Show result
    st.image(image, caption="Processed Image", use_column_width=True)
    st.success("✅ Helmet detection completed!")

    # Delete temporary file
    os.remove(temp_file_path)
