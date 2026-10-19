from flask import Flask, render_template, request, make_response
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io
import base64
import csv

app = Flask(__name__)

MODEL_PATH = Path(
    "runs/detect/runs/detect/yolo11n_paint_scratch_trial-2/weights/best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH.resolve()}"
    )

model = YOLO(str(MODEL_PATH))

# Class-specific confidence thresholds
CLASS_THRESHOLDS = {
    "crack": 0.20,
    "dent": 0.20,
    "missing-head": 0.25,
    "paint-off": 0.25,
    "scratch": 0.25,
}


def filter_missing_head_duplicates(result):
    """Remove highly overlapping nested missing-head predictions."""
    boxes = result.boxes

    if boxes is None or len(boxes) < 2:
        return result

    coords = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()

    keep = []

    # Process higher-confidence predictions first
    for i in sorted(
        range(len(scores)),
        key=lambda k: scores[k],
        reverse=True
    ):
        duplicate = False

        if model.names[int(classes[i])] == "missing-head":
            x1, y1, x2, y2 = coords[i]
            area_i = max(0, x2 - x1) * max(0, y2 - y1)

            for j in keep:
                if model.names[int(classes[j])] != "missing-head":
                    continue

                a1, b1, a2, b2 = coords[j]

                ix1 = max(x1, a1)
                iy1 = max(y1, b1)
                ix2 = min(x2, a2)
                iy2 = min(y2, b2)

                intersection = (
                    max(0, ix2 - ix1) *
                    max(0, iy2 - iy1)
                )

                area_j = max(0, a2 - a1) * max(0, b2 - b1)
                smaller_area = min(area_i, area_j)

                if (
                    smaller_area > 0
                    and intersection / smaller_area >= 0.80
                ):
                    duplicate = True
                    break

        if not duplicate:
            keep.append(i)

    result.boxes = boxes[keep]
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    uploaded_file = request.files.get("image")

    if not uploaded_file or not uploaded_file.filename:
        return "Please upload an image.", 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")

        # Run YOLO prediction
        results = model.predict(
            source=image,
            conf=0.10,
            iou=0.45,
            max_det=100,
            agnostic_nms=False,
            verbose=False
        )

        result = results[0]

        # Remove nested duplicate missing-head predictions
        result = filter_missing_head_duplicates(result)

        # Apply class-specific confidence thresholds
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            keep = []

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                threshold = CLASS_THRESHOLDS.get(
                    class_name, 0.25
                )

                if confidence >= threshold:
                    keep.append(i)

            result.boxes = boxes[keep]

        # Prepare detection details
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            coordinates = [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ]

            detections.append({
                "name": class_name,
                "confidence": round(confidence * 100, 2),
                "coordinates": coordinates
            })

        print("\n--- FINAL DETECTIONS ---")

        for detection in detections:
            print(
                f"Class: {detection['name']}, "
                f"Confidence: {detection['confidence']}%, "
                f"Coordinates: {detection['coordinates']}"
            )

        # Generate annotated image
        plotted_image = result.plot()
        output_image = Image.fromarray(
            plotted_image[:, :, ::-1]
        )

        buffer = io.BytesIO()
        output_image.save(buffer, format="JPEG")

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # Generate CSV report in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        writer.writerow([
            "Defect Class",
            "Confidence (%)",
            "X1",
            "Y1",
            "X2",
            "Y2"
        ])

        for detection in detections:
            writer.writerow([
                detection["name"],
                detection["confidence"],
                *detection["coordinates"]
            ])

        report_data = base64.b64encode(
            csv_buffer.getvalue().encode("utf-8")
        ).decode("utf-8")

        return render_template(
            "result.html",
            image_data=encoded_image,
            detections=detections,
            report_data=report_data
        )

    except Exception:
        app.logger.exception("Image detection failed")
        return (
            "Image processing failed. Check the VS Code terminal.",
            500
        )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)