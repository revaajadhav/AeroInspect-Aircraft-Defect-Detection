# AeroInspect — Project Methodology

## 1. Project Overview

AeroInspect is a computer vision prototype designed to identify visible aircraft surface defects in images. It uses a trained YOLO object detection model and a Flask web application to display predictions and generate inspection reports.

## 2. System Workflow

The system follows these stages:

1. **Image Upload:** The user uploads an aircraft surface image through the web interface.
2. **Image Preparation:** The uploaded image is opened using Pillow and converted to RGB format.
3. **Object Detection:** The trained YOLO model processes the image and predicts defect classes, confidence scores, and bounding-box coordinates.
4. **Duplicate Filtering:** Nested predictions for the `missing-head` class are filtered when their overlap with a higher-confidence prediction reaches the configured threshold.
5. **Confidence Filtering:** Class-specific confidence thresholds are applied to remove predictions below the selected minimum score.
6. **Image Annotation:** Bounding boxes and labels are drawn on the image using the YOLO plotting function.
7. **Results Display:** The Flask application displays the annotated image, defect counts, and confidence scores.
8. **Report Generation:** The application provides a downloadable annotated image and a CSV report containing defect classes, confidence scores, and bounding-box coordinates.

## 3. Defect Categories

The model is configured for the following five classes:

- Crack
- Dent
- Missing-head
- Paint-off
- Scratch

## 4. Model Inference and Filtering

The application loads the trained YOLO weights from the path configured in `app.py`.

The current inference settings include:

- Initial confidence threshold: 0.10
- IoU setting: 0.45
- Maximum detections: 100
- Class-specific confidence thresholds:
  - Crack: 0.20
  - Dent: 0.20
  - Missing-head: 0.25
  - Paint-off: 0.25
  - Scratch: 0.25

The class-specific thresholds are applied after inference. These values are configuration settings and do not represent the model's measured accuracy.

For `missing-head` predictions, the application compares bounding boxes and removes a lower-confidence prediction when at least 80% of the smaller box's area overlaps a retained higher-confidence box of the same class.

## 5. Web Application

The application uses Flask to handle image uploads and return the detection results page.

The interface includes:
- An image-upload page.
- An annotated image preview.
- Total detection and unique-class counts.
- Mean prediction confidence.
- A summary of detections for each configured class.
- A list of individual predictions.
- A download option for the annotated image.
- A downloadable CSV report.

## 6. Technologies

- **Python:** Application logic and image processing workflow.
- **Ultralytics YOLO:** Object detection inference.
- **Pillow:** Opening and converting uploaded images.
- **Flask:** Web server and request handling.
- **HTML and CSS:** User interface.
- **CSV:** Exporting prediction details.

## 7. Evaluation and Limitations

The confidence score describes the model's score for a prediction; it is not automatically a calibrated probability that a defect is real.

The current web application demonstrates the inference and reporting workflow. Its user-interface tests do not, by themselves, establish detection accuracy.

Formal model evaluation should use a separate labelled test dataset and report suitable metrics such as precision, recall, and mean Average Precision (mAP). Actual results should be included only after they have been verified.

The system may produce false positives or miss real defects. It is a research prototype and is not certified for aircraft airworthiness decisions.

## 8. Future Work

- Evaluate performance on a labelled test dataset.
- Analyse false positives and missed defects.
- Test robustness across different aircraft images and lighting conditions.
- Improve duplicate filtering using further validation.
- Expand the documentation with verified experimental results.
