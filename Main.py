import streamlit as st
from PIL import Image
import os
from utils import ImageSense  # Make sure the class name matches your utils.py

# Initialize the agent
@st.cache_resource(show_spinner=False)
def load_agent():
    return ImageSense()

agent = load_agent()

st.set_page_config(page_title="ImageSense", layout="wide")
st.title("ImageSense: Multi-Task Visual Intelligence Tool")
st.markdown(
    "Upload an image and select a task: **Captioning** (BLIP), **Object Detection** (YOLOv5n), or **OCR** (EasyOCR)."
)

task_dic = {
    "Caption": "caption",
    "Object Detection": "object_detection",
    "OCR": "ocr"
}

with st.sidebar:
    st.subheader("Settings")
    task_option = st.selectbox("Select task", list(task_dic.keys()), index=0)
    task_prompt = task_dic[task_option]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
    run_btn = st.button("Run Now")

with col2:
    st.subheader("Output")
    if run_btn and uploaded_file and image is not None:
        if task_prompt == "caption":
            with st.spinner("Generating caption..."):
                caption = agent.run_caption(image)
            st.write("**Caption:**", caption)
        elif task_prompt == "object_detection":
            with st.spinner("Running object detection..."):
                det_result = agent.run_object_detection(image)
                det_img = agent.plot_detection(image, det_result)
            st.image(det_img, caption="Detection Result")
        elif task_prompt == "ocr":
            with st.spinner("Running OCR..."):
                ocr_result = agent.run_ocr(image)
                ocr_img = agent.plot_ocr(image, ocr_result)
            st.image(ocr_img, caption="OCR Result")
            for bbox, text, conf in ocr_result:
                st.write(f"**Text:** {text} (Confidence: {conf:.2f})")

