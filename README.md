# AeroInspect — Aircraft Surface Defect Detection

## Overview
AeroInspect is a computer vision research prototype for detecting aircraft surface defects from images using a YOLO object detection model. The application provides an annotated image, predicted defect classes, confidence scores, and a downloadable CSV inspection report.

## Objectives
- Detect visible aircraft surface defects in uploaded images.
- Display predicted defect locations using bounding boxes.
- Show class-specific confidence scores.
- Provide downloadable annotated images and CSV reports.

## Defect Classes
The model is configured to detect five defect categories:
- Crack
- Dent
- Missing-head
- Paint-off
- Scratch

## Technologies Used
- Python
- YOLO and Ultralytics
- Flask
- Pillow
- HTML and CSS
- CSV report generation

## Application Features
- Upload an aircraft surface image.
- Run object detection using the trained model.
- View the annotated output image.
- Review defect counts and confidence scores.
- Download the annotated image.
- Download the inspection results as a CSV file.

## How to Run

1. Install the required Python packages:

   ```bash
   pip install flask ultralytics pillow
   ```

2. Ensure the trained model is available at the path configured in `app.py`.

3. Open a terminal in the project directory and run:

   ```bash
   python app.py
   ```

4. Open the following address in your browser:

   `http://127.0.0.1:5000/`

5. Upload an image to view the detection results.

## Project Status
The Flask-based prototype is implemented, and the image detection and report-download workflow has been tested.

## Limitations
Predictions may contain false positives or missed defects. Confidence scores are model scores, not guaranteed probabilities of correctness. The prototype is intended for research and demonstration; it is not certified for aircraft airworthiness decisions and does not replace qualified inspection.

## Future Improvements
- Evaluate the model using precision, recall, and mAP on a labelled test dataset.
- Improve detection performance across different lighting conditions and aircraft surfaces.
- Expand testing on unseen images and defect types.
- Improve inspection reporting and project documentation.
