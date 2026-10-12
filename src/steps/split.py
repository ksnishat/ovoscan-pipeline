
from typing import Annotated

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step


@step
def split_data(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "val_df"]
]:
    """
    Splits the dataset into training and validation sets (80/20 split).
    """
    try:
        # Check if we have enough data to split
        if len(df) < 10:
            raise ValueError("Not enough data to split!")

        # Stratified split ensures class balance is maintained
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )

        print(" Data Split Complete.")
        print(f"   Train Set: {len(train_df)} images")
        print(f"   Val Set:   {len(val_df)} images")

        return train_df, val_df

    except Exception as e:
        print(f" Error in data splitting: {e}")
        raise e
