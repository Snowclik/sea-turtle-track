"""
src/scripts/split_dataset.py
Divide las imágenes originales en Train (80%), Val (10%) y Test (10%).
Garantiza el aislamiento absoluto para la investigación.
"""

import os
import glob
import random
import shutil

def split_dataset(base_dir: str, seed: int = 42):
    random.seed(seed)
    
    orig_images_dir = os.path.join(base_dir, "dataset", "original")
    orig_labels_dir = os.path.join(base_dir, "dataset", "labels")
    
    splits_dir = os.path.join(base_dir, "dataset", "splits")
    
    # Crear estructura base
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(splits_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(splits_dir, split, "labels"), exist_ok=True)
        
    images = sorted(glob.glob(os.path.join(orig_images_dir, "*.*")))
    
    # Filtrar solo archivos de imagen
    valid_ext = ('.jpg', '.jpeg', '.png')
    images = [img for img in images if img.lower().endswith(valid_ext)]
    
    random.shuffle(images)
    
    total = len(images)
    # 80/10/10 split
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    
    # Asegurar que al menos haya 1 en val y test si el dataset es muy pequeño
    if len(val_imgs) == 0 and len(train_imgs) > 1:
        val_imgs = [train_imgs.pop()]
    if len(test_imgs) == 0 and len(train_imgs) > 1:
        test_imgs = [train_imgs.pop()]
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    
    for split_name, img_list in splits.items():
        for img_path in img_list:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(orig_labels_dir, f"{stem}.txt")
            
            # Copiar imagen
            shutil.copy2(img_path, os.path.join(splits_dir, split_name, "images", os.path.basename(img_path)))
            
            # Copiar label (si existe)
            if os.path.isfile(label_path):
                shutil.copy2(label_path, os.path.join(splits_dir, split_name, "labels", os.path.basename(label_path)))
                
        print(f"[{split_name.upper()}] Copiadas {len(img_list)} imagenes.")
        
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    split_dataset(base)
