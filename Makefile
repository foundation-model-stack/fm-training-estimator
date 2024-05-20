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

.PHONY: hook
hook:
	pre-commit install --hook-type commit-msg