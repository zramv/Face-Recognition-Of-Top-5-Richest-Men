# Face Recognition Of Top 5 Richest Men


## Project Overview

This project is a face recognition system built using **Python** and the **InsightFace** library. It is designed to detect and identify the faces of the **top 5 richest people in the world**. The system can process both **pre-recorded videos** and **real-time** camera streams, making it suitable for live demonstrations or offline analysis.

The core functionality relies on the **InsightFace** framework, which is known for its high-accuracy face recognition capabilities. One of the main reasons for selecting InsightFace is its support for **GPU acceleration**, which significantly improves performance compared to traditional CPU-based processing. This enables the system to handle real-time face detection and recognition efficiently, especially when working with high-resolution inputs.

## Features
- **Face Detection and Recognition**<br>
    Accurately detects and identifies faces using the InsightFace library.
- **GPU Acceleration**<br>
    Utilizes GPU for faster processing and improved performance compared to CPU-only solutions.
- **High Accuracy Recognition**  
    Based on state-of-the-art deep learning models provided by InsightFace.


  <br>
  <br>
<details>
<summary>System Configration</summary> 
  
- #### Python Packages

1. OpenCV - Computer vision library<br>
   ```pip install opencv-python```
2. InsightFace - Face recognition using ONNX<br>
   ```pip install insightface```
3. ONNXRuntime GPU - To run AI models on GPU<br>
   ```pip install onnxruntime-gpu```
4. scikit-learn - For cosine similarity<br>
   ```pip install scikit-learn```


- #### NVIDIA Software
1. NVIDIA Driver  
   - Required to use the GPU  
   - Check with: nvidia-smi

2. CUDA Toolkit
3. cuDNN
</details>
  <br>


## Top 5 Richest Men
<p align="center">
  <img src="known_faces/Elon Musk/image2.jpg" alt="Elon Musk" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="known_faces/Bernard Arnault/image2.jpg" alt="Bernard Arnault" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="known_faces/Jeff Bezos/image2.jpg" alt="Jeff Bezos" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="known_faces/Larry Ellison/image1.jpg" alt="Larry Ellison" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="known_faces/Mark Zuckerberg/image1.jpg" alt="Mark Zuckerberg" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
<p align="center">
  <strong>Elon Musk</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Bernard Arnault</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Jeff Bezos</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Larry Ellison</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Mark Zuckerberg</strong>&nbsp;
</p>

## How It Works (Video)
https://github.com/user-attachments/assets/beec239c-ed8b-448b-b59d-bcbd1e961c71.mp4

https://github.com/user-attachments/assets/c795c205-86fe-4553-a850-2c7452cc8b88.mp4




