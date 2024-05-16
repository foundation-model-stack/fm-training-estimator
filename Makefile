.PHONY: build
build: lint fmt install
	tox -e build

.PHONY: install
install:
	pip install -e .

.PHONY: fmt
fmt:
	tox -e fmt

.PHONY: lint
lint:
	tox -e lint
