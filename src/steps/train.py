import os
import shutil
import pandas as pd
from zenml import step
from ultralytics import YOLO
from typing_extensions import Annotated

def create_yolo_structure(df: pd.DataFrame, split_name: str, root_dir: str):
    """
    Helper: Creates symlinks to organize data for YOLO training.
    Format: root_dir/split_name/label/image.jpg
    """
    for _, row in df.iterrows():
        src_path = row['image_path']
        label = row['label']
        
        # Define target directory (e.g., /tmp/yolo/train/fertile)
        target_dir = os.path.join(root_dir, split_name, label)
        os.makedirs(target_dir, exist_ok=True)
        
        # Create Symlink
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, file_name)
        
        # Only link if it doesn't exist (prevents errors on re-runs)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except OSError:
                # Fallback for Windows or filesystems that block symlinks
                shutil.copy(src_path, dst_path)

@step
def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 16
) -> Annotated[YOLO, "trained_model"]:
    """
    Trains a YOLOv8 Classification model using the ingested data.
    """
    try:
        # 1. Setup Temporary Training Workspace
        # We use a local cache folder so we don't mess up our raw data
        yolo_root = os.path.abspath("yolo_dataset_cache")
        
        # Clean previous runs to ensure fresh start
        if os.path.exists(yolo_root):
            shutil.rmtree(yolo_root)
            
        print(f" Preparing YOLO data structure in {yolo_root}...")
        create_yolo_structure(train_df, "train", yolo_root)
        create_yolo_structure(val_df, "val", yolo_root)
        
        # 2. Initialize Model (Transfer Learning)
        # We use 'yolov8n-cls.pt' (Nano) for speed. 
        print(" Initializing YOLOv8 Nano model...")
        model = YOLO('yolov8n-cls.pt') 
        
        # 3. Train
        print(f" Starting Training for {epochs} epochs on GPU...")
        # device=0 forces NVIDIA GPU usage
        results = model.train(
            data=yolo_root,
            epochs=epochs,
            imgsz=224,
            batch=batch_size,
            device=0,           
            project="ovoscan_train_runs",
            name="experiment_1",
            exist_ok=True       
        )
        
        print(f" Training Complete. Top1 Accuracy: {results.top1}")
        return model

    except Exception as e:
        print(f" Training Failed: {e}")
        raise e