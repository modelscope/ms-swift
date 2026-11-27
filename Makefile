WHL_BUILD_DIR :=package
DOC_BUILD_DIR :=docs/build/

# default rule
default: whl docs

.PHONY: docs
docs:
	bash .dev_scripts/build_docs.sh

.PHONY: linter
linter:
	ruff check .
	ruff format --check .

.PHONY: test
test:
	pytest -m "not slow"

.PHONY: whl
whl:
	python -m build

.PHONY: clean
clean:
	rm -rf  $(WHL_BUILD_DIR) $(DOC_BUILD_DIR)
