# ImageSense: Multi-Task Visual Intelligence Tool

**Note:** This project is designed to be lightweight and compatible with low-end devices and PCs. All models and dependencies are chosen for efficient performance on systems with limited resources.

ImageSense is a multi-task visual intelligence tool that allows you to perform several vision tasks on images, including:

- **Image Captioning**
- **Object Detection** 
- **Optical Character Recognition (OCR)**

## Features

- Upload an image and select a task from the sidebar.
- Instantly get results for captioning, object detection, or OCR.
- Simple, interactive interface powered by Streamlit.

## Coming Soon

- Visual Question Answering (VQA)
- Image Segmentation

## Requirements

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [Pillow](https://python-pillow.org/)
- [easyocr](https://github.com/JaidedAI/EasyOCR)
- [YOLOv5n weights (`yolov5n.pt`)](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt) (for object detection)
- Other dependencies as listed in `requirements.txt`

## Setup

1. **Clone the repository** and navigate to the project folder:
    ```sh
    git clone <repo-url>
    cd Image_sense/ImageSense
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download YOLOv5n weights**  
   Place `yolov5n.pt` in the `ImageSense` directory for object detection to work.

4. **Run the tool:**
    ```sh
    streamlit run Main.py
    ```
    or, if `streamlit` is not recognized:
    ```sh
    python -m streamlit run Main.py
    ```

5. **Open your browser** and go to [http://localhost:8501](http://localhost:8501)



