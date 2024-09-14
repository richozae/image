import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from streamlit_lottie import st_lottie
import requests
import io

# Fungsi untuk memuat model YOLO dari file .pt
@st.cache_resource
def load_yolo_model(model_path):
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error("Model file not found. Please check the path.")
        return None

# Fungsi untuk mendeteksi produk dan menggambar bounding box
def detect_and_draw(image, model):
    image_np = np.array(image)
    results = model(image_np)
    detection_data = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # confidence scores
        labels = result.boxes.cls.cpu().numpy()  # class labels

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Bounding box color
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 20)
            label_text = f'{model.names[int(label)]}'
            cv2.putText(image_np, label_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)

            detection_data.append({
                'label': model.names[int(label)],
                'confidence': score,
                'box': (x1, y1, x2, y2)
            })
    
    return Image.fromarray(image_np), detection_data

# Fungsi untuk memuat animasi Lottie
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Streamlit app
st.set_page_config(page_title="Product Detection with AI", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")

# Tambahkan animasi Lottie di bagian header
lottie_ai = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_z9ed2jna.json")

with st.container():
    st_lottie(lottie_ai, height=300, key="AI")
    st.title("Product Detection with AI")
    st.write("Project By : RICHO")
    st.write("---")
    st.subheader("Analyze your product images using YOLO AI technology.")
    st.write("Upload an image and press **Process** to see the results!")

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
st.sidebar.markdown("Explore the options below:")

# Load model
model_path = 'best.pt'
model = load_yolo_model(model_path)

if model is not None:
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process 🚀'):
            st.write("Processing...")

            # Progress bar
            progress_bar = st.progress(0)

            # Detect and draw bounding box
            for percent in range(100):
                progress_bar.progress(percent + 1)

            processed_image, detection_data = detect_and_draw(image, model)
            
            # Display output image
            st.image(processed_image, caption='Processed Image', use_column_width=True)
            
            # Display results in a table
            if detection_data:
                st.write("Detected Products:")
                results_table = {}
                for data in detection_data:
                    label = data['label']
                    if label in results_table:
                        results_table[label]['count'] += 1
                        results_table[label]['confidences'].append(data['confidence'])
                    else:
                        results_table[label] = {
                            'count': 1,
                            'confidences': [data['confidence']]
                        }

                # Convert results_table to a displayable format
                display_data = {
                    'Product': [],
                    'Accuracy': [],
                    'Count': []
                }

                for label, info in results_table.items():
                    display_data['Product'].append(label)
                    # Calculate average confidence and format it as percentage
                    avg_confidence = np.mean(info['confidences']) * 100
                    display_data['Accuracy'].append(f"{avg_confidence:.2f}%")
                    display_data['Count'].append(info['count'])

                # Display the data as a table
                st.table(display_data)

                # Convert the processed image to a downloadable format
                img_byte_arr = io.BytesIO()  # create a buffer for saving the image
                processed_image.save(img_byte_arr, format='PNG')  # save image to the buffer
                img_byte_arr = img_byte_arr.getvalue()  # get the byte data of the image

                # Add download button for the image
                st.download_button(
                    label="Download Processed Image",
                    data=img_byte_arr,
                    file_name="processed_image.png",
                    mime="image/png"
                )
            else:
                st.write("No products detected.")
else:
    st.write("Please provide a valid model path.")

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    <div class="footer">
        Made with 💙 by Richo | © 2024
    </div>
""", unsafe_allow_html=True)
