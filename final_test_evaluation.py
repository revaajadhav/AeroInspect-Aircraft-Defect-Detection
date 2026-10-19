
from pathlib import Path
from ultralytics import YOLO

DATA = "FINAL_DATASET/data.yaml"
TEST_IMAGES = Path("FINAL_DATASET/test/images")

models = {
    "Baseline": "runs/detect/runs/yolo11n_baseline/weights/best.pt",
    "Paint-Scratch": (
        "runs/detect/runs/detect/"
        "yolo11n_paint_scratch_trial-2/weights/best.pt"
    ),
}

# Check test images
if not TEST_IMAGES.is_dir():
    raise SystemExit(
        f"Test images folder not found: {TEST_IMAGES.resolve()}"
    )

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_count = sum(
    1 for p in TEST_IMAGES.iterdir()
    if p.is_file() and p.suffix.lower() in valid_extensions
)

print(f"Test images found: {image_count}")

if image_count == 0:
    raise SystemExit("No test images found. Evaluation cancelled.")

# Confirm paths
data_path = Path(DATA).resolve()
test_path = data_path.parent / "test" / "images"

print("DATA YAML:", data_path)
print("TEST PATH:", test_path.resolve())
print("TEST IMAGE COUNT:", image_count)

# Evaluate saved models
for name, weights in models.items():
    weights_path = Path(weights)

    if not weights_path.is_file():
        print(f"\nSKIPPING {name}: weights not found: {weights_path}")
        continue

    print(f"\n===== {name} TEST RESULTS =====")

    model = YOLO(str(weights_path))

    metrics = model.val(
    data=str(Path(DATA).resolve()),
    split="test",
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    plots=True,
    project="runs/test_evaluation",
    name=name.lower().replace("-", "_"),
    exist_ok=True,
    verbose=True
)

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

print("\nEvaluation finished.")
