import torch
import cv2
import blobfile as bf
from PIL import Image
import numpy as np
import os
import tqdm

def unfold_img(img, size):
    """Unfolds img list of patches."""

    img_patches = img.unfold(0, size, size).unfold(1, size, size).unfold(2, 3, 3)
    print(img_patches.shape)
    img_patches = img_patches.reshape(-1, size, size, 3)
    return img_patches

def read_img(img_fn, resize_to=None):
    with bf.BlobFile(img_fn, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
    pil_image = pil_image.convert("RGB")
    if resize_to is not None:
        pil_image = pil_image.resize(resize_to)
    return pil_image



ROOT = "/cluster/scratch/jminder/RoadSegmentation/data/massachusetts-roads/tiff/"
OUTDIR = "/cluster/scratch/jminder/RoadDiffusion/data/massachusetts-roads/"
SIZE = 256

os.makedirs(OUTDIR)
train_files = ["train/" + file for file in os.listdir(ROOT + "train") if file.endswith("tiff")]
val_files = ["val/" + file for file in os.listdir(ROOT + "val") if file.endswith("tiff")]
test_files = ["test/" + file for file in os.listdir(ROOT + "test") if file.endswith("tiff")]

all_files = train_files + val_files + test_files

for file in tqdm.tqdm(all_files):
    im = read_img(ROOT + file, (1792,1792))
    im = np.array(im)
    n = im.shape[0]//SIZE
    all_white_sum = SIZE * SIZE * 255 * 3
    all_white_limit = all_white_sum * 0.6
    patches = []
    for i in range(n):
        for j in range(n):
            patch = im[i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE, :]
            if patch.sum() < all_white_limit:
                outpath = OUTDIR + file.split("/")[1] + f"-{i}-{j}.jpg"
                Image.fromarray(patch).save(outpath)
