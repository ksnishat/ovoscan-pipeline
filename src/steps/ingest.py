import os
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple, List

# Define the mapping logic here.
# We keep raw data in 3 folders, but train on 2 classes initially.
CLASS_MAPPING = {
    "fertile": "fertile",
    "infertile": "defect",
    "dead": "defect"
}

@step
def ingest_data(data_dir: str = "data/raw") -> Annotated[pd.DataFrame, "dataset"]:
    """
    Ingests data from the raw data directory and creates a DataFrame.
    
    Args:
        data_dir: Path to the raw data folder containing subfolders 
                  (fertile, infertile, dead).
                  
    Returns:
        pd.DataFrame: A dataframe containing 'image_path' and 'label'.
    """
    try:
        images: List[str] = []
        labels: List[str] = []
        
        print(f"🔍 Scanning data directory: {data_dir}")
        
        # We look for the folder names present in our mapping keys
        for folder_name, target_label in CLASS_MAPPING.items():
            folder_path = os.path.join(data_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Warning: Folder '{folder_name}' not found. Skipping.")
                continue
                
            # List all valid images
            files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   Found {len(files)} images in '{folder_name}' -> Mapped to '{target_label}'")
            
            for filename in files:
                # Store absolute path to be safe
                full_path = os.path.abspath(os.path.join(folder_path, filename))
                images.append(full_path)
                labels.append(target_label)
                
        # Create DataFrame
        df = pd.DataFrame({"image_path": images, "label": labels})
        
        # Basic Validation
        if df.empty:
            raise ValueError("No images found! Check data/raw structure.")
            
        print(f"Ingestion Complete. Total images: {len(df)}")
        print(f"   Class Distribution: {df['label'].value_counts().to_dict()}")
        
        return df

    except Exception as e:
        print(f"Error in data ingestion: {e}")
        raise e