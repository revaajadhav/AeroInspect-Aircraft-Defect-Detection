
from pathlib import Path
import cv2

import matplotlib.pyplot as plt

IMAGE_DIR = Path("FINAL_DATASET/train/images")
OUTPUT_DIR = Path("image_processing_preview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
)

if not images:
    print("No images found:", IMAGE_DIR.resolve())
else:
    image_path = images[0]
    original = cv2.imread(str(image_path))

    if original is None:
        print("Could not read:", image_path)
    else:
        # Convert to grayscale and apply CLAHE contrast enhancement
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)

        cv2.imwrite(
            str(OUTPUT_DIR / "original_gray.png"), gray
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "clahe_enhanced.png"), enhanced
        )

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap="gray")
        plt.title("Original grayscale")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced, cmap="gray")
        plt.title("CLAHE contrast enhancement")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            str(OUTPUT_DIR / "clahe_comparison.png"),
            dpi=150
        )
        plt.show()

        print("Saved comparison in:", OUTPUT_DIR.resolve())
