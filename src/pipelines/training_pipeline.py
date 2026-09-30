from zenml import pipeline
from src.steps.ingest import ingest_data
from src.steps.split import split_data
from src.steps.train import train_model

@pipeline
def ovoscan_training_pipeline(
    data_path: str,
    epochs: int = 5,
    batch_size: int = 16
):
    """
    Orchestrates the End-to-End Training Flow.
    
    Flow:
    1. Ingest Data (Raw Folders -> DataFrame)
    2. Split Data (DataFrame -> Train/Val)
    3. Train Model (Train/Val -> YOLO Model)
    """
    # 1. Ingest
    df = ingest_data(data_dir=data_path)
    
    # 2. Split
    train_df, val_df = split_data(df=df)
    
    # 3. Train
    model = train_model(
        train_df=train_df, 
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size
    )