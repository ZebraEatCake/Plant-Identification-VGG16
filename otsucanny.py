from pathlib import Path
import cv2

# ------------------------------------------------------------------
# 1) Collect images
# ------------------------------------------------------------------
INPUT_ROOT  = Path("Dataset (disease)")
OUTPUT_ROOT = Path("Dataset (disease)_otsucanny")

image_paths = list(INPUT_ROOT.rglob("*.jpg"))     # recursive search
images      = []
valid_paths = []

for p in image_paths:
    img = cv2.imread(str(p))
    if img is not None:
        images.append(img)
        valid_paths.append(p)
    else:
        print(f"[WARN] Could not read: {p}")

# ------------------------------------------------------------------
# 2) Process with Otsu threshold + Canny combined
# ------------------------------------------------------------------
for orig_path, img in zip(valid_paths, images):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu Thresholding to determine the optimal threshold
    otsu_thresh_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Canny using Otsu as low threshold and a multiplier (e.g. 2x) as high
    low_thresh  = int(otsu_thresh_val * 0.5)
    high_thresh = int(otsu_thresh_val * 1.5)
    edges       = cv2.Canny(blurred, low_thresh, high_thresh)

    # Output path
    rel_path = orig_path.relative_to(INPUT_ROOT)
    out_dir  = OUTPUT_ROOT / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_name = f"{orig_path.stem}_otsucanny{orig_path.suffix}"
    cv2.imwrite(str(out_dir / combined_name), edges)
    print(f"Saved ➜ {out_dir / combined_name}")
