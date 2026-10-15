"""Entry point for running the OvoScan training pipeline as a script.

ZenML needs a real file to determine the source root: running the pipeline via
``python -c`` fails with "Unable to determine source root because the main
module does not have an associated file". The Airflow DAG therefore invokes this
script as a subprocess rather than importing the pipeline directly, which also
isolates ZenML's logging from Airflow's (the two otherwise form a cycle that
recurses until Python hits its recursion limit).
"""

import argparse
import os
import sys

# Allow `python src/run_training.py` from the repository root as well as from
# inside a container where the tree is mounted at /opt/airflow.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.training_pipeline import ovoscan_training_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OvoScan training pipeline")
    parser.add_argument(
        "--data-path",
        default=os.getenv("OVOSCAN_DATA_DIR", "egg_dataset/train"),
        help="Directory containing the YOLO images/ and labels/ split",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("OVOSCAN_EPOCHS", "3")))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("OVOSCAN_BATCH_SIZE", "8"))
    )
    args = parser.parse_args()

    print(f"Starting OvoScan training pipeline on {args.data_path}")
    ovoscan_training_pipeline(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
