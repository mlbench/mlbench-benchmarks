
TAG:=latest
DOCKER_REGISTRY:=hub.docker.com

.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ; ## remove all build, test, coverage and Python artifacts


docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

publish-pytorch-base: ## Build and publish the pytorch base image. Usage: make publish-pytorch-base
	docker build -t mlbench-pytorch-base:$(TAG) ./pytorch/base/
	docker tag mlbench-pytorch-base:$(TAG) $(DOCKER_REGISTRY)/mlbench-pytorch-base:$(TAG)
	docker push $(DOCKER_REGISTRY)/mlbench-pytorch-base:$(TAG)

publish-tensorflow-base: ## Build and publish the tensorflow base image. Usage: make publish-tensorflow-base
	docker build -t mlbench-tensorflow-base:$(TAG) ./tensorflow/base/
	docker tag mlbench-tensorflow-base:$(TAG) $(DOCKER_REGISTRY)/mlbench-tensorflow-base:$(TAG)
	docker push $(DOCKER_REGISTRY)/mlbench-tensorflow-base:$(TAG)

