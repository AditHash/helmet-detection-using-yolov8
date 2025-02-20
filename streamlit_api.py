import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load trained YOLO model
MODEL_PATH = r"C:\helmet\best.pt"
model = YOLO(MODEL_PATH)

st.title("Helmet Detection App")
st.write("Upload an image, and the model will detect if helmets are worn or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Save image temporarily
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
    cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Run YOLO inference
    results = model(temp_image_path)
    
    # Draw bounding boxes with thin green lines
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            
            # Draw thin green box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Display the result
    st.image(image, caption="Processed Image", use_column_width=True)
    st.write("Confidence Scores:")
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])  # Confidence score
            st.write(f"{conf:.2f}")
    
    # Remove temporary file
    os.remove(temp_image_path)
