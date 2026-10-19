from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\Aircraft_oops_cp")

DATASETS = {
    "ASDD": BASE / "ASDD",

    "IISc": BASE / "IISc",

    "Innovation": BASE / "Innovation Hangar v2.v1-aug.yolov8",

    "AircraftSkinV1": BASE / "Aircraft_Skin_Defects_v1",
}

OUTPUT = BASE / "Aircraft project" / "FINAL_DATASET"


# ============================================================
# COMMON CLASSES
# ============================================================

CLASS_NAMES = [
    "crack",
    "dent",
    "missing-head",
    "paint-off",
    "scratch"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# STATISTICS
# ============================================================

stats = {
    name: {
        "images": 0,
        "labels": 0,
        "detections": 0,
        "segmentations_converted": 0,
        "empty_skipped": 0,
        "invalid_skipped": 0,
    }
    for name in DATASETS
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CONVERT ONE LABEL FILE
# ============================================================

def convert_label(label_path):

    output_lines = []

    with open(label_path, "r", encoding="utf-8") as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Empty label
    if not lines:
        return None, "empty"

    for line in lines:

        parts = line.split()

        # Need at least class + coordinates
        if len(parts) < 5:
            return None, "invalid"

        try:
            class_id = int(float(parts[0]))
            values = list(map(float, parts[1:]))

        except ValueError:
            return None, "invalid"

        # ----------------------------------------------------
        # CLASS CHECK
        # ----------------------------------------------------

        if class_id < 0 or class_id >= NUM_CLASSES:
            return None, "invalid"

        # ----------------------------------------------------
        # STANDARD YOLO BOUNDING BOX
        # class x y w h
        # ----------------------------------------------------

        if len(values) == 4:

            x, y, w, h = values

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):
                return None, "invalid"

            output_lines.append(
                f"{class_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

        # ----------------------------------------------------
        # YOLO SEGMENTATION
        # class x1 y1 x2 y2 ...
        # ----------------------------------------------------

        elif len(values) >= 6 and len(values) % 2 == 0:

            xs = values[0::2]
            ys = values[1::2]

            # Coordinates must be normalized
            if not all(0 <= x <= 1 for x in xs):
                return None, "invalid"

            if not all(0 <= y <= 1 for y in ys):
                return None, "invalid"

            # Bounding rectangle around polygon
            xmin = min(xs)
            xmax = max(xs)

            ymin = min(ys)
            ymax = max(ys)

            width = xmax - xmin
            height = ymax - ymin

            if width <= 0 or height <= 0:
                return None, "invalid"

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            output_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        else:

            return None, "invalid"

    return output_lines, "valid"


# ============================================================
# PROCESS DATASETS
# ============================================================

for dataset_name, dataset_root in DATASETS.items():

    print("\n" + "=" * 70)
    print("PROCESSING:", dataset_name)
    print("=" * 70)

    for split in ["train", "valid", "test"]:

        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        if not image_dir.exists() or not label_dir.exists():

            print(
                f"{split}: folder missing - skipped"
            )

            continue

        image_files = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:

            label_path = label_dir / (
                image_path.stem + ".txt"
            )

            # ------------------------------------------------
            # NO LABEL → SKIP
            # ------------------------------------------------

            if not label_path.exists():

                continue

            # ------------------------------------------------
            # CONVERT LABEL
            # ------------------------------------------------

            converted, status = convert_label(label_path)

            if status == "empty":

                stats[dataset_name]["empty_skipped"] += 1
                continue

            if status == "invalid" or converted is None:

                stats[dataset_name]["invalid_skipped"] += 1
                continue

            # ------------------------------------------------
            # UNIQUE FILE NAME
            # ------------------------------------------------

            unique_name = (
                dataset_name
                + "_"
                + image_path.name
            )

            output_image = (
                OUTPUT
                / split
                / "images"
                / unique_name
            )

            output_label = (
                OUTPUT
                / split
                / "labels"
                / (
                    dataset_name
                    + "_"
                    + image_path.stem
                    + ".txt"
                )
            )

            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                output_image
            )

            # ------------------------------------------------
            # SAVE STANDARD YOLO LABEL
            # ------------------------------------------------

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(converted)
                )

            # ------------------------------------------------
            # UPDATE STATS
            # ------------------------------------------------

            stats[dataset_name]["images"] += 1
            stats[dataset_name]["labels"] += 1

            for line in converted:

                stats[dataset_name]["detections"] += 1

            # Count original segmentation files
            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                original_lines = [
                    x.strip()
                    for x in f
                    if x.strip()
                ]

            for line in original_lines:

                if len(line.split()) > 5:

                    stats[dataset_name][
                        "segmentations_converted"
                    ] += 1


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_content = f"""path: {OUTPUT.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: {NUM_CLASSES}

names:
"""

for i, name in enumerate(CLASS_NAMES):

    yaml_content += f"  {i}: {name}\n"


with open(
    OUTPUT / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_content)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n")
print("=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

for dataset_name, s in stats.items():

    print("\n" + dataset_name)

    print("Images copied              :", s["images"])
    print("Labels created             :", s["labels"])
    print("Annotations                :", s["detections"])
    print(
        "Segmentation annotations  :",
        s["segmentations_converted"]
    )
    print(
        "Empty labels skipped       :",
        s["empty_skipped"]
    )
    print(
        "Invalid labels skipped     :",
        s["invalid_skipped"]
    )


print("\n" + "=" * 70)
print("FINAL DATASET LOCATION")
print(OUTPUT)
print("=" * 70)

print("\nClasses:")

for i, name in enumerate(CLASS_NAMES):

    print(i, "=", name)