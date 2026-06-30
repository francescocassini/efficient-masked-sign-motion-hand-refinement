PYTHON ?= python
DATASET ?= phoenix
IMAGE ?= efficient-masked-sign-motion:local

.PHONY: models data train-p3 infer-p3 infer-p5 smoke docker-build docker-models docker-data docker-train-p3 docker-infer-p3 docker-infer-p5

models:
	$(PYTHON) scripts/download_models.py

data:
	bash scripts/download_dataset_from_hf.sh "$${SOKENAR_DATA_REPO}" "$${SOKENAR_DATA_ROOT:-datasets}"

train-p3:
	$(PYTHON) -m train --cfg configs/train/p3_csl_phoenix.yaml --nodebug --use_gpus 0 --device 0 --num_nodes 1

infer-p3:
	$(PYTHON) -m test --cfg configs/infer/p3_$(DATASET).yaml --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1

infer-p5:
	$(PYTHON) -m test --cfg configs/infer/p5_$(DATASET).yaml --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1

smoke:
	$(PYTHON) -m compileall -q mGPT scripts tests
	$(PYTHON) tests/test_release_layout.py

docker-build:
	docker build -t $(IMAGE) .

docker-models:
	docker compose run --rm sokenar models

docker-data:
	docker compose run --rm sokenar data

docker-train-p3:
	docker compose run --rm sokenar train-p3

docker-infer-p3:
	docker compose run --rm sokenar infer-p3 $(DATASET)

docker-infer-p5:
	docker compose run --rm sokenar infer-p5 $(DATASET)
