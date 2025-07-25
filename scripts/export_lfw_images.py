from sklearn.datasets import fetch_lfw_people
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

# Configuration
SAVE_ROOT = Path("data/raw/images")
IMAGES_PER_PERSON = 5
N_PERSONS = 10

# Download in memory
lfw = fetch_lfw_people(download_if_missing=True, resize=1.0, color=True)
names = lfw.target_names
targets = lfw.target
images = lfw.images

SAVE_ROOT.mkdir(parents=True, exist_ok=True)

count_by_person = {name: 0 for name in names}
used_persons = []

for i in range(len(images)):
    name = names[targets[i]]
    if count_by_person[name] < IMAGES_PER_PERSON:
        person_dir = SAVE_ROOT / name.replace(" ", "_")
        person_dir.mkdir(parents=True, exist_ok=True)
        img_arr = images[i].astype(np.uint8)
        # If image is grayscale, convert to 3-channel for saving as jpg
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        out_path = person_dir / f"{count_by_person[name]:03d}.jpg"
        imageio.imwrite(out_path, img_arr)
        count_by_person[name] += 1
        if name not in used_persons and len(used_persons) < N_PERSONS:
            used_persons.append(name)
    # Stop when done
    if len(used_persons) == N_PERSONS and all(count_by_person[n] >= IMAGES_PER_PERSON for n in used_persons):
        break

print(f"Exported {IMAGES_PER_PERSON} images for {N_PERSONS} persons to {SAVE_ROOT}")
