IMAGE ?= icr.io/ftplatform/fm_training_estimator:latest

.PHONY: build
build: lint fmt install
	tox -e build

.PHONY: install
install: hook
	pip install -e .

.PHONY: fmt
fmt:
	tox -e fmt

.PHONY: lint
lint:
	tox -e lint

.PHONY: test
test:
	tox -e test

.PHONY: hook
hook:
	pre-commit install --hook-type commit-msg

.PHONY: build-model
build-model:
	python -m fm_training_estimator.regressor.xgboost.train ./workdir/data.csv ./workdir/model.json '["tokens_per_second","memory","memory_act"]'

.PHONY: run-web-ui
run-web-ui:
	python -m fm_training_estimator.ui.web ./workdir/model_whitelist.txt ./workdir/data.csv ./workdir/model.json --use_model_features=True --enable_api=True

.PHONY: run-cli
run-cli:
	python -m fm_training_estimator.ui.cli -l ./workdir/data.csv -m ./workdir/model.json $(CONF)

.PHONY: run-api
run-api:
	python -m fm_training_estimator.ui.api ./workdir/data.csv ./workdir/model.json --use_model_features=True

.PHONY: cbuild
cbuild:
	docker build -t ${IMAGE} -f Dockerfile .

.PHONY: cpush
cpush:
	docker push ${IMAGE}
