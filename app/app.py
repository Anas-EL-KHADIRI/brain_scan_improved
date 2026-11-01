import base64
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.title("BrainScan AI")

mode = st.radio("Mode", ["Classification", "Detection"])
uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    st.image(Image.open(io.BytesIO(image_bytes)), caption="Uploaded MRI", width=300)

    if st.button("Predict"):
        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/jpeg")}

        try:
            if mode == "Classification":
                res = requests.post(f"{API_URL}/api/v1/classify", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.subheader(f"Prediction: {data['predicted_class']}")
                    st.write(f"Confidence: {data['confidence_pct']:.2f}%")
                    st.bar_chart(data["probabilities"])
                else:
                    st.error(f"Error {res.status_code}: {res.text}")

            elif mode == "Detection":
                res = requests.post(f"{API_URL}/api/v1/detect", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.subheader(f"Detections: {data['num_detections']}")
                    if data.get("annotated_image_base64"):
                        img_data = base64.b64decode(data["annotated_image_base64"])
                        st.image(Image.open(io.BytesIO(img_data)), caption="Detected Tumors", width=400)
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("API offline. Run in another terminal: uvicorn api.main:app --port 8000 --reload")
