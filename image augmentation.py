from imgaug import augmenters as iaa
from pathlib import Path
import glob, cv2

#Input image file path
INPUT_ROOT  = Path("Dataset_otsucanny")
OUTPUT_ROOT = Path("Dataset_otsucanny_aug2")

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


#  Save original images to output directory
for orig_path, img in zip(valid_paths, images):
    rel_path = orig_path.relative_to(INPUT_ROOT)
    out_dir  = OUTPUT_ROOT / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / rel_path.name
    cv2.imwrite(str(out_path), img)
    print(f"Saved original ➜ {out_path}")


# Define augmenters
augmenters = [
    ("rot90",  iaa.Rotate(90)),     # rotate +90°
    ("rot180", iaa.Rotate(180)),    # rotate 180°
    ("fliplr", iaa.Fliplr(1.0))     # horizontal flip
]


# Loop over each augmenter, apply, and save
for suffix, aug in augmenters:
    aug_imgs = aug(images=images)   # batch-augment

    for orig_path, aug_img in zip(valid_paths, aug_imgs):
        rel_path  = orig_path.relative_to(INPUT_ROOT)          # keep sub-folders
        out_dir   = OUTPUT_ROOT / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        new_name  = f"{orig_path.stem}_{suffix}{orig_path.suffix}"
        out_path  = out_dir / new_name

        cv2.imwrite(str(out_path), aug_img)
        print(f"Saved augmented ➜ {out_path}")
