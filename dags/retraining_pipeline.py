# FILE: dags/retraining_pipeline.py
"""Scheduled retraining for the OvoScan egg classifier.

Flow: retrain via the ZenML pipeline -> reload the serving model -> report drift.

Every project import is inside the task callables. A module-level import of
ultralytics or zenml pulls in torch, and if that import fails the DAG file fails
to parse and the DAG silently disappears from the Airflow UI. Importing lazily
means a dependency problem surfaces as a failed task with a traceback instead.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to path so Airflow can find the project modules. This must point
# outside /opt/airflow/dags: Airflow scans that directory for DAG files, so
# mounting src inside it makes Airflow try to parse every project module as a
# DAG and report import errors.
#
# /opt/airflow is also added because this project's modules import each other
# with the absolute form "from src.steps...", which requires the parent of src
# to be importable.
sys.path.append(os.getenv("OVOSCAN_SRC", "/opt/airflow/src"))
sys.path.append(os.getenv("OVOSCAN_ROOT", "/opt/airflow"))


def retrain_model(**context):
    """Run the ZenML training pipeline in a subprocess.

    ZenML logs through the root logger, and Airflow's logging_mixin re-logs
    anything written to its stream back to that same logger. The two form a
    cycle that recurses until Python hits its recursion limit, so the task dies
    before training starts. Running the pipeline in a separate process gives it
    its own logging state and breaks the cycle.

    This is also the right shape for a real training job: the heavy work is
    isolated from the scheduler process, so a crash or an OOM cannot take the
    scheduler down with it.
    """
    import subprocess

    data_path = os.getenv("OVOSCAN_DATA_DIR", "/opt/airflow/dataset/train")
    epochs = os.getenv("OVOSCAN_EPOCHS", "3")
    batch_size = os.getenv("OVOSCAN_BATCH_SIZE", "8")
    src_dir = os.getenv("OVOSCAN_SRC", "/opt/airflow/src")
    root_dir = os.getenv("OVOSCAN_ROOT", "/opt/airflow")

    # A real script file, not `python -c`: ZenML needs a file to determine the
    # source root and fails with "Unable to determine source root because the
    # main module does not have an associated file" otherwise.
    result = subprocess.run(
        [
            "python",
            os.path.join(src_dir, "run_training.py"),
            "--data-path",
            data_path,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
        ],
        capture_output=True,
        text=True,
        cwd=root_dir,
        timeout=int(os.getenv("OVOSCAN_TRAIN_TIMEOUT", "3600")),
    )

    # Surface the tail of the output so a failure is diagnosable from the task
    # log rather than only from the subprocess's exit code.
    tail = (result.stdout or "")[-2000:]
    if result.returncode != 0:
        raise RuntimeError(
            f"Training subprocess failed (exit {result.returncode}).\n"
            f"--- stdout tail ---\n{tail}\n"
            f"--- stderr tail ---\n{(result.stderr or '')[-2000:]}"
        )

    return {
        "status": "trained",
        "data_path": data_path,
        "epochs": int(epochs),
        "ds": context["ds"],
        "stdout_tail": tail[-500:],
    }


def reload_serving_model(**context):
    """Tell the serving API to pick up the freshly retrained weights.

    Best-effort by design: if the API is not running (for example when only the
    DAG is being exercised) the task logs that and succeeds. The point is to
    make the retrain -> serve handoff visible in the DAG graph and to fail
    loudly when the API is reachable but rejects the new weights.
    """
    import json
    import urllib.error
    import urllib.request

    url = os.getenv(
        "OVOSCAN_API_RELOAD_URL", "http://host.docker.internal:8003/admin/reload-model"
    )

    try:
        request = urllib.request.Request(url, method="POST", data=b"")
        # The serving API enforces X-API-Key on /admin/reload-model. Without
        # this header the call returns 401, which is not in the skip list
        # below, so the task raised and the whole DAG failed.
        api_key = os.getenv("API_KEY")
        if api_key:
            request.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {"status": "reloaded", "response": payload}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:200]
        if exc.code in (403, 404):
            # Reload is disabled or not implemented, not broken.
            return {"status": "skipped", "reason": f"HTTP {exc.code}: {detail}"}
        raise RuntimeError(f"API rejected the reload ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        return {"status": "skipped", "reason": f"API unreachable: {exc.reason}"}


def report_drift(**context):
    """Read the serving API's drift state so the run records it."""
    import json
    import urllib.error
    import urllib.request

    url = os.getenv("OVOSCAN_MONITORING_URL", "http://host.docker.internal:8003/monitoring")

    # /monitoring enforces X-API-Key. Without the header this returned 401 and
    # the task silently reported "skipped", so the DAG recorded no drift state.
    request = urllib.request.Request(url)
    api_key = os.getenv("API_KEY")
    if api_key:
        request.add_header("X-API-Key", api_key)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {
            "status": "reported",
            "drift_psi": payload.get("drift_psi"),
            "drift_status": payload.get("drift_status"),
        }
    except urllib.error.HTTPError as exc:
        # A 401/403 means we were rejected, not that the API is absent. That is
        # a misconfiguration and must fail the run rather than pass silently.
        if exc.code in (401, 403):
            raise RuntimeError(
                f"drift endpoint rejected the request (HTTP {exc.code}). "
                "Check that API_KEY matches the serving API."
            ) from exc
        return {"status": "skipped", "reason": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"status": "skipped", "reason": f"API unreachable: {exc.reason}"}


default_args = {
    "owner": "nishat",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ovoscan_continuous_learning",
    default_args=default_args,
    description="Weekly YOLOv8-cls retraining + model reload + drift report",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["ovoscan", "mlops"],
) as dag:
    train_task = PythonOperator(
        task_id="retrain_classifier",
        python_callable=retrain_model,
    )

    reload_task = PythonOperator(
        task_id="reload_serving_model",
        python_callable=reload_serving_model,
    )

    drift_task = PythonOperator(
        task_id="report_drift",
        python_callable=report_drift,
    )

    train_task >> reload_task >> drift_task
