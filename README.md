# OvoScan: Intelligent Quality Control System

![Python](https://img.shields.io/badge/Python-3.10-00599C?style=flat&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Microservices-2496ED?style=flat&logo=docker&logoColor=white) ![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-blue) ![Llama 3](https://img.shields.io/badge/GenAI-Llama3-orange) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**OvoScan** is an automated computer vision system for industrial hatchery safety. It detects egg defects (cracks, infertility) using a custom-trained **YOLOv8** model and employs a **Generative AI Agent (Llama 3)** to generate ISO-compliant technical reports in English/German.

---

## 🚀 Key Features
* **🔍 Custom Vision:** Fine-tuned YOLOv8 model (`model.pt`) optimized for organic surface defects.
* **🧠 AI Safety Consultant:** Local **Llama 3** agent (via Ollama) that analyzes detection data and writes maintenance recommendations.
* **🏗️ Microservices Architecture:** Fully containerized setup with separate Backend (FastAPI), Frontend (Streamlit), and AI Engine (Ollama).
* **⚡ Production Ready:** Uses **Data Version Control (DVC)** for reproducibility and **GPU Passthrough** for high-speed inference.

---

## 🛠️ Tech Stack
* **AI:** YOLOv8, Ollama + Llama 3
* **Orchestration:** Docker Compose (Host Networking)
* **Backend:** FastAPI + Uvicorn
* **Frontend:** Streamlit
* **MLOps:** DVC (Data Version Control)

---

## ⚡ Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/ksnishat/ovoscan-pipeline.git
cd ovoscan-pipeline
```

2. **Pull the Dataset (DVC):**
```bash
# Downloads the specific image version used for the 99.9% accuracy model
dvc pull
```

3. **Start Services (Docker Compose):**
```bash
# Uses 'network_mode: host' for low-latency GPU access
sudo docker compose up -d
```

4. **Initialize the Brain:**
```bash
# Download the Llama 3 model into the running container
sudo docker exec -it ovoscan_ollama ollama pull llama3
```

5. **Open the Apps:**
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8001/docs

---

## 🏗️ Architecture (High Level)

| Stage | Description |
|-------|-------------|
| **Input** | Image uploaded via Streamlit UI. |
| **Detection** | API runs YOLOv8 inference and returns bounding boxes/confidence scores. |
| **Agent** | Detection metrics are passed to the Llama 3 container to evaluate severity against the "Hatchery Manual" (RAG). |
| **Output** | A technical report is displayed in the UI justifying the rejection decision. |

---

## ⚙️ Engineering Challenges & Solutions

Recovery steps taken during the development of the Edge Architecture:

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Container Networking** | API could not reach Ollama via localhost in Bridge mode. | Host Networking: Switched to `network_mode: "host"` to unify the network stack for low-latency communication. |
| **GPU Invisibility** | Docker containers defaulted to CPU, causing slow inference. | NVIDIA Runtime: Configured `deploy.resources` in Docker Compose to explicitly reserve GPU capabilities. |
| **Data Tracking** | Difficulty tracking which images improved model accuracy. | DVC: Implemented Data Version Control to hash-track the dataset versions. |

---

## 📧 Contact

Developed by Khaled. For collaboration or questions, open an issue.