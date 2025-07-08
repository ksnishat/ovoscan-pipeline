from src.pipelines.training_pipeline import ovoscan_training_pipeline
import os

if __name__ == "__main__":
    # Point to the local data/raw folder
    # We use absolute path to be safe
    data_path = os.path.abspath("data/raw")
    
    print(f" Starting OvoScan Training Pipeline using data at: {data_path}")
    
    ovoscan_training_pipeline(
        data_path=data_path,
        epochs=3,       # Low epochs for the first test run
        batch_size=8    # Safe for RTX 3050 Ti
    )