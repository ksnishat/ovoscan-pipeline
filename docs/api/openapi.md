# OvoScan API Documentation

> OpenAPI 3.0 Specification for OvoScan Computer Vision Backend API

## Base URL

```
http://localhost:8000
```

## Endpoints

### `GET /`

Root endpoint.

**Response:**
```json
{
  "service": "OvoScan API",
  "version": "1.0.0",
  "description": "Manufacturing Quality Inspection API"
}
```

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-10-02T10:30:00.000Z"
}
```

---

### `POST /detect`

Detect manufacturing defects in an image using YOLOv8.

**Request (Multipart Form):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (JPG/PNG/TIFF, max 10MB) |
| `conf` | float | No | Confidence threshold (default: 0.25) |
| `iou` | float | No | IoU threshold for NMS (default: 0.45) |
| `classes` | string | No | Comma-separated class IDs to filter |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `detections` | array[object] | List of detected defects |
| `num_detections` | integer | Total detections |
| `processing_time_ms` | float | Processing time in milliseconds |
| `annotated_image` | string | Base64 encoded image with boxes |
| `model_version` | string | Model version used |

**Detection Object:**
| Field | Type | Description |
|-------|------|-------------|
| `class_name` | string | Defect class |
| `class_id` | integer | Class ID |
| `confidence` | float | Confidence score |
| `bbox` | array[float] | Bounding box [x1, y1, x2, y2] |
| `segmentation` | array | Segmentation mask (if applicable) |

---

### `POST /quality`

Full quality assessment with defect classification and severity scoring.

**Request:** Same as `/detect`

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `overall_quality_score` | float | 0.0 to 1.0 quality score |
| `overall_quality_label` | string | "pass", "warning", or "fail" |
| `defect_summary` | object | Summary by defect class |
| `severity_breakdown` | object | Severity counts |
| `recommendations` | array[string] | Action recommendations |
| `ai_report` | string | LLM-generated quality report |
| `processing_time_ms` | float | Total processing time |

**Defect Summary Object:**
```json
{
  "crack": {"count": 5, "avg_confidence": 0.92},
  "scratch": {"count": 2, "avg_confidence": 0.87}
}
```

**Severity Breakdown:**
```json
{
  "critical": 2,
  "major": 3,
  "minor": 5,
  "none": 190
}
```

---

### `POST /report`

Generate AI-powered quality report with RAG integration.

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file |
| `language` | string | No | Report language ("en"/"de") |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `report` | string | Generated report |
| `language` | string | Report language |
| `overall_quality` | string | Quality rating |
| `defect_summary` | object | Summary statistics |
| `recommendations` | array[string] | Action items |
| `processing_time_ms` | float | Processing time |

---

### `POST /pipeline`

Run full ZenML pipeline (detection + quality assessment + report).

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file |
| `generate_report` | boolean | No | Generate LLM report (default: true) |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_id` | string | Pipeline run ID |
| `status` | string | Pipeline execution status |
| `results` | object | Combined results from all steps |
| `processing_time_ms` | float | Total pipeline execution time |

---

### `GET /models`

List registered models.

**Response:**
```json
[
  {"name": "ovoscan-defect-model", "version": "1.0.0", "stage": "Production"},
  {"name": "ovoscan-quality-model", "version": "1.0.0", "stage": "Staging"}
]
```

---

### `GET /docs` and `GET /openapi.json`

Interactive API documentation.

## Metrics

- `ovoscan_detection_seconds` - Defect detection latency histogram
- `ovoscan_defects_total` - Total defects counter (by class)
- `ovoscan_quality_score` - Quality score gauge
- `ovoscan_pipeline_duration_seconds` - ZenML pipeline duration
- `ovoscan_api_requests_total` - Total API request counter