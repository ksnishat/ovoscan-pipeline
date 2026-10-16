.PHONY: help install dev-install test lint format clean dvc-run dvc-repro dvc-push dvc-pull docker-build docker-up docker-down k8s-deploy k8s-undeploy zenml-run stack stack-stop tls load-test

## Show this help message
help:
	@echo "OvoScan Pipeline - Available commands:"
	@echo ""
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //' | column -t -s ':'

## Install production dependencies
install:
	pip install -r requirements.txt

## Install development dependencies
dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff mypy pre-commit

## Run tests
test:
	python -m pytest tests/ -v --tb=short --maxfail=3

## Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml

## Lint with ruff
lint:
	ruff check src/ tests/

## Format code with ruff
format:
	ruff format src/ tests/

## Clean up generated files
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf yolo_dataset_cache/

## Run DVC pipeline (full)
dvc-run:
	dvc repro

## Reproduce DVC pipeline from scratch
dvc-repro:
	dvc repro --force

## Push data to DVC remote
dvc-push:
	dvc push

## Pull data from DVC remote
dvc-pull:
	dvc pull

## Build Docker image
docker-build:
	docker build -t ovoscan/api:latest -f Dockerfile .

## Start all services with Docker Compose
docker-up:
	docker compose up -d

## Stop all services
docker-down:
	docker compose down

## Deploy to Kubernetes
k8s-deploy:
	helm install ovoscan ./helm-chart

## Undeploy from Kubernetes
k8s-undeploy:
	helm uninstall ovoscan

## Run ZenML pipeline
zenml-run:
	python -m src.pipelines.training_pipeline

## Start the live stack (API, MLflow, MinIO, Prometheus, Grafana)
stack:
	./run_stack.sh

## Stop the live stack and its containers
stack-stop:
	@-pkill -f "run_api.py" 2>/dev/null || true
	@-docker rm -f prom_ovoscan graf_ovoscan minio_ovoscan 2>/dev/null || true
	@echo "stopped"

## Start the HTTPS reverse proxy (self-signed, port 8446)
tls:
	./infrastructure/tls/start_tls_proxy.sh

## Run the load test against the live stack
load-test:
	python3 load_test.py --requests 60 --concurrency 6
