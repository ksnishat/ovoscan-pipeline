# OvoScan — Operations Reference

Setup, configuration and troubleshooting. The [README](../README.md) covers what the project is and how it scores.

## Running Tests
```bash
uv run --frozen --extra dev python -m pytest tests/ -v   # 173 passed, 1 skipped, 9 deselected
```

The 10 skipped tests are integration tests that need the live docker stack. They
are excluded from CI (which has no model checkpoint or dataset) and run against
real services instead of an in-process `TestClient`:

```bash
./run_stack.sh
LIVE_STACK=1 pytest tests/test_integration_live.py -v   # 9 passed
```

They assert the API classifies a real image, that `/analyze-report` still returns
a report when Ollama is down (template fallback, not a 500), that Prometheus is
actually scraping the API (not merely healthy), and that MLflow, Grafana and
MinIO are reachable.

## Monitoring & Live Demo
```bash
./run_stack.sh                   # API + MLflow + MinIO + Prometheus + Grafana
python3 provision_dashboards.py ovoscan   # datasource + dashboard
./verify_stack.sh                 # 7 checks, exit 0 when all pass
```

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI** | http://localhost:8003/docs | N/A |
| **Prometheus** | http://localhost:9093 | N/A |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **MLflow** | http://localhost:5000 | N/A |
| **Streamlit** | http://localhost:8504 | N/A |

> **Running the stack:** `./run_stack.sh` starts this project's API, MLflow, MinIO, Prometheus and Grafana. Dashboard: `streamlit run src/dashboard.py` (port 8504). Airflow and the feature store run through `docker compose`. The four-project orchestrator is kept in the author's workspace and is not part of this repository.
| **ZenML** | http://localhost:8080 | N/A |

### Exposed metrics
| Metric | Type | Meaning |
|--------|------|---------|
| `ovoscan_predictions_total` | counter | Predictions, labelled by `class_label` |
| `ovoscan_prediction_confidence` | histogram | Confidence distribution |
| `ovoscan_prediction_latency_seconds` | histogram | Inference latency |
| `ovoscan_model_loaded` | gauge | 1 when the model is loaded, 0 otherwise |
| `ovoscan_inference_errors_total` | counter | Inference failures |
| `ovoscan_class_distribution` | gauge | Rolling class distribution |
| `ovoscan_prediction_drift_psi` | gauge | PSI of the live prediction distribution vs the validation baseline |
| `ovoscan_drift_status` | gauge | 0 stable, 1 moderate, 2 significant, -1 unknown |

### Drift detection
`ovoscan_class_distribution` is a rolling counter, not a drift metric: it has no
baseline to compare against, so it cannot detect drift on its own. PSI is now
computed against a reference distribution and exposed as
`ovoscan_prediction_drift_psi`. `GET /monitoring` reports `drift_psi`,
`drift_status` and the baseline in use.

The baseline defaults to a balanced fertile/infertile split and is overridable
via `set_drift_baseline` so a retrained model can ship its own reference.

Verified discrimination: a balanced stream gives PSI 0.0 (stable) while a
95%-fertile stream gives 1.33 (significant).

### Scheduled retraining
`dags/retraining_pipeline.py` runs weekly: retrain via the ZenML pipeline →
reload the serving model → report drift.

```bash
docker compose up -d airflow-postgres airflow-webserver airflow-scheduler
docker exec ovoscan_airflow_scheduler airflow dags unpause ovoscan_continuous_learning
docker exec ovoscan_airflow_scheduler airflow dags trigger ovoscan_continuous_learning
```

Airflow UI: http://localhost:8084 (`admin` / `admin`).

The pipeline runs in a subprocess (`src/run_training.py`). ZenML logs through the
root logger and Airflow's `logging_mixin` re-logs its own stream, which forms a
cycle that recurses until Python hits its recursion limit. A subprocess gives
the pipeline its own logging state and also isolates heavy training from the
scheduler process.

## RAG Quality Assistant
`src/agent/rag.py` implements a **ChromaDB + LangChain** retrieval pipeline over
a hatchery quality-control manual, using `sentence-transformers/all-MiniLM-L6-v2`
embeddings and a local Ollama LLM.

The knowledge base ships with the repository at
`knowledge_base/hatchery_manual.txt`, so `rag_available` is **`true`** out of the
box. The API reports it in the health response:

```bash
curl http://localhost:8003/
# {"status":"running","service":"ovoscan-ai-v2","model_loaded":true,"rag_available":true,"device":"cuda"}
```

### How it works
1. The manual is split with `RecursiveCharacterTextSplitter` (800 chars, 120
   overlap), splitting on section boundaries first so the disposition tables
   stay intact.
2. Chunks are embedded and persisted to a ChromaDB collection at
   `data/chroma/` — built once, reused across restarts.
3. A defect query retrieves the top 4 chunks, which are injected into a
   criteria / action / escalation prompt.
4. The local LLM answers **only** from the retrieved context.

### Verified behaviour
Asking about an infertile egg returns:

```
1. CRITERIA - The manual does not specify the criteria that identify an
   infertile egg (it only lists visual indicators for fertile eggs).
2. ACTION - Remove the egg from the incubation stream immediately and record
   the source flock identifier and collection date.
3. ESCALATION - If the infertility rate for a single flock exceeds 8 percent
   over a rolling 7-day window.
```

The 8 percent threshold appears in Section 6.1 of the manual and nowhere in the
prompt — so this is genuine retrieval, not the model reciting training data.

### Configuration
| Variable | Default | Purpose |
|----------|---------|---------|
| `KNOWLEDGE_BASE_PATH` | `knowledge_base/hatchery_manual.txt` | Manual to index |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Vector store location |
| `CHROMA_COLLECTION` | `hatchery_rules` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `OLLAMA_MODEL` | `llama3.2` | Generation model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

To index your own manual instead:

```bash
KNOWLEDGE_BASE_PATH=/path/to/your/manual.txt python -m src.agent.rag
```

> **Hardware note:** a 4 GB GPU cannot hold a 26B model. Use a small model
> (1B-3B) for responsive RAG, or accept CPU-bound latency with a larger one.

### Measuring retrieval quality
Retrieval that *runs* and retrieval that *finds the right passage* are different
claims, and only the second one is testable. `src/agent/rag_eval.py` scores the
retriever against a labelled set of 10 questions. Each case names the section
that should be retrieved and evidence tokens that must appear in the retrieved
text — tokens chosen so a passage from the wrong section cannot satisfy them by
accident. `"8 percent"` alone matches several sections of the manual, so the
infertility case also requires `"Flock Health"`.

```bash
uv run --frozen python src/agent/rag_eval.py                  # report and gate
uv run --frozen python src/agent/rag_eval.py --json           # machine-readable
uv run --frozen python src/agent/rag_eval.py --min-recall 0.9  # stricter gate
```

Current result: `recall@4 = 0.900`, MRR 0.783, `hit@1` 0.700 across 10 cases.

Retrieval metrics need only the embedding model and the index, so they run
offline and deterministically and are gated in CI (`recall@4 >= 0.8`).
Generation metrics are deliberately *not* gated: they need an LLM, a CI runner
has none, and a faithfulness score that silently reports 0.0 when the model is
absent is worse than no score.

**The one failure is real, not a test artefact.** The Section 5 disposition
table is intact in a single chunk — the code comment claiming it is preserved is
accurate — but that chunk also carries the `SECTION 6` heading and `6.1`, which
dilutes its embedding. Asked *"what happens when a fertile egg also has a shell
defect?"*, the chunk holding the answer ranks **7th of 11** and is not retrieved
at k=4. At a larger k it would be found. The fix is section-boundary chunking;
it is documented here rather than quietly raising k, because raising k hides the
problem instead of measuring it.

## Kubernetes Deployment
```bash
helm install ovoscan ./helm-chart
# or
kubectl apply -f k8s/
```

## Project Structure
```plaintext
ovoscan-pipeline/
├── dags/
├── docs/
├── helm-chart/
├── infrastructure/
├── k8s/
├── src/
│   ├── agent/
│   ├── dash_app/
│   ├── pipelines/
│   ├── scripts/
│   ├── utils/
│   ├── __init__.py
│   ├── app.py
│   ├── auth.py
│   ├── config.py
│   ├── dashboard.py
│   ├── ratelimit.py
│   ├── retry.py
│   ├── run_training.py
│   ├── uploads.py
│   └── versioning.py
├── terraform/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.serve
├── dvc.yaml
├── load_test.py
├── Makefile
├── params.yaml
├── provision_dashboards.py
├── pyproject.toml
├── README.md
├── requirements-serve.txt
├── run.py
├── run_stack.sh
├── uv.lock
└── verify_stack.sh
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| **YOLO model not found** | Train with the command above and copy `best.pt` to `models/ovoscan_cls_best.pt` |
| **`docker compose up` → missing `Dockerfile.mlflow`** | Fixed in this commit — the file is now tracked in git |
| **Tests fail with "model not loaded"** | `TestClient(app)` must be used as a context manager so the FastAPI lifespan runs: `with TestClient(app) as c:` |
| **ZenML pipeline signature introspection fails** | ZenML 0.97 wraps the pipeline; read `pipeline.entrypoint` for the real signature |
| **`rag_available: false`** | The knowledge base is missing or the RAG deps failed to import. Check `knowledge_base/hatchery_manual.txt` exists and that `chromadb`, `langchain-chroma` and `langchain-huggingface` are installed |
| **RAG responses are very slow** | A large Ollama model is spilling to CPU. Check `ollama ps` — if the PROCESSOR column shows a high CPU percentage, switch to a smaller model via `OLLAMA_MODEL` |
| **ZenML stack not registered** | Run `zenml stack register` with the correct orchestrator/artifact/metadata stores |
| **DVC remote not accessible** | Check the DVC remote storage configuration (S3/GCS/Azure) |
| **Ollama connection refused** | Verify Ollama is running on port 11434 |
| **Grafana shows "No data"** | The provisioning or dashboards directory is not mounted, or Grafana is not on the `mlops-stack` network so it cannot resolve `prom_<project>` | Run `python3 provision_dashboards.py` — it reports which of the two it is |
| **GPU not detected in K8s** | Install the NVIDIA device plugin and ensure GPU nodes are available |
