# Lane Detection System for Autonomous Vehicles

This project implements a real-time **Lane Detection System** using Computer Vision techniques. It detects lane lines on a road video, calculates the average slope for left/right lanes, and visualizes the driving path, mimicking the fundamental perception layer of self-driving cars.

## Tech Stack
* **Language:** Python 3.12
* **Library:** OpenCV (Computer Vision), NumPy (Matrix Operations)

## Methodology
The pipeline consists of the following engineering steps:
1.  **Gray Scaling:** Conversion to grayscale to reduce processing load.
2.  **Noise Reduction:** Applying Gaussian Blur (Kernel: 5x5) to smooth out image noise.
3.  **Edge Detection:** Using **Canny Edge Detector** to identify high-frequency gradients.
4.  **ROI Masking:** Defining a triangular Region of Interest to isolate the road surface.
5.  **Hough Transform:** Converting edge pixels into mathematical line segments ($y=mx+b$).
6.  **Data Smoothing:** Calculating average slope and intercept to generate a single, stable line for each lane.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt