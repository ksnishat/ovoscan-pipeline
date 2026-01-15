import os
import shutil
import pandas as pd
from zenml import step
from ultralytics import YOLO
from typing_extensions import Annotated

# ... (Keep create_yolo_structure helper exactly as is) ...
def create_yolo_structure(df: pd.DataFrame, split_name: str, root_dir: str):
    # ... (No changes here) ...
    for _, row in df.iterrows():
        src_path = row['image_path']
        label = row['label']
        target_dir = os.path.join(root_dir, split_name, label)
        os.makedirs(target_dir, exist_ok=True)
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, file_name)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except OSError:
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
        yolo_root = os.path.abspath("yolo_dataset_cache")
        
        # We DO NOT clean previous runs here anymore to avoid race conditions
        # But we ensure the dataset folder structure is fresh
        dataset_path = os.path.join(yolo_root, "dataset")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            
        print(f" Preparing YOLO data structure in {dataset_path}...")
        create_yolo_structure(train_df, "train", dataset_path)
        create_yolo_structure(val_df, "val", dataset_path)
        
        # 2. Initialize Model
        print("🤖 Initializing YOLOv8 Nano model...")
        model = YOLO('yolov8n-cls.pt') 
        
        # 3. Train
        # We define explicit project/name paths so we know where to find the weights later
        project_path = os.path.join(yolo_root, "ovoscan_train_runs")
        run_name = "experiment_1"
        
        print(f" Starting Training for {epochs} epochs on GPU...")
        model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=224,
            batch=batch_size,
            device=0,           
            project=project_path,
            name=run_name,
            exist_ok=True       
        )
        
        # 4. CRITICAL FIX: Reload the clean model from disk
        # This removes the un-pickleable DataLoaders
        best_weights_path = os.path.join(project_path, run_name, "weights", "best.pt")
        print(f"💾 Loading best weights from {best_weights_path} for artifact storage...")
        
        clean_model = YOLO(best_weights_path)
        
        return clean_model

    except Exception as e:
        print(f" Training Failed: {e}")
        raise e