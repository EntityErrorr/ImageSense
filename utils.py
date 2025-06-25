import streamlit as st
from PIL import Image
import numpy as np
import os

# BLIP for image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

# YOLOv5n for object detection
from yolov5 import YOLOv5

# EasyOCR for OCR
import easyocr

# Initialize BLIP (captioning)
@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

blip_processor, blip_model = load_blip()

# Initialize YOLOv5n (object detection)
YOLOV5_WEIGHTS = "yolov5n.pt"
@st.cache_resource(show_spinner=False)
def load_yolo():
    if os.path.exists(YOLOV5_WEIGHTS):
        return YOLOv5(YOLOV5_WEIGHTS, device="cpu")
    return None

yolo_detector = load_yolo()

# Initialize EasyOCR
@st.cache_resource(show_spinner=False)
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)

ocr_reader = load_easyocr()

class ImageSense:
    def run_caption(self, image):
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def run_object_detection(self, image):
        if yolo_detector is None:
            return []
        results = yolo_detector.predict(image)
        return results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    def plot_detection(self, image, det_result):
        import cv2
        img = np.array(image)
        for det in det_result:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{int(cls)}:{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        return img

    def run_ocr(self, image):
        result = ocr_reader.readtext(np.array(image))
        return [(bbox, text, conf) for (bbox, text, conf) in result]

    def plot_ocr(self, image, ocr_result):
        import cv2
        img = np.array(image)
        for bbox, text, conf in ocr_result:
            pts = np.array(bbox).astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        return img

agent = ImageSense()

if "process" not in st.session_state:
    st.session_state["process"] = False

class StreamlitApp:
    def __init__(self):
        self.task_dic = {
            "Caption": "caption",
            "Object Detection": "object_detection",
            "OCR": "ocr"
        }
        self.task_prompt = None
        st.set_page_config(page_title="ImageSense", layout="wide")
        st.title("ImageSense: Multi-Task Visual Intelligence Tool")
        st.markdown(
            "Upload an image and select a task: Captioning (BLIP), Object Detection (YOLOv5n), or OCR (EasyOCR)."
        )

    def run(self):
        col1, col2 = st.columns([1, 2])

        with st.sidebar:
            st.subheader("Settings")
            task_option = st.selectbox("Select task", list(self.task_dic.keys()), index=0)
            self.task_prompt = self.task_dic[task_option]

        with col1:
            st.subheader("Input")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            image = None
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image")
            if st.button("Run Now"):
                st.session_state["process"] = True

        with col2:
            st.subheader("Output")
            if "process" in st.session_state and st.session_state["process"]:
                if uploaded_file and image is not None:
                    if self.task_prompt == "caption":
                        caption = agent.run_caption(image)
                        st.write("Caption:", caption)
                    elif self.task_prompt == "object_detection":
                        if yolo_detector is None:
                            st.warning("YOLOv5n weights 'yolov5n.pt' not found. Please download and place them in the working directory.")
                        else:
                            det_result = agent.run_object_detection(image)
                            det_img = agent.plot_detection(image, det_result)
                            st.image(det_img, caption="Detection Result")
                    elif self.task_prompt == "ocr":
                        ocr_result = agent.run_ocr(image)
                        ocr_img = agent.plot_ocr(image, ocr_result)
                        st.image(ocr_img, caption="OCR Result")
                        for bbox, text, conf in ocr_result:
                            st.write(f"Text: {text} (Confidence: {conf:.2f})")
                st.session_state["process"] = False

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    app = StreamlitApp()
    app.run()


    