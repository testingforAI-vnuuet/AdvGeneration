.PHONY: help prepare-dev test lint run doc

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3

PROJECT_DIR = $(shell pwd)
PROJECT_NAME = AdvGeneration

run:
	$(VENV_ACTIVATE) && export PYTHONPATH=$(PROJECT_DIR)/src:$PYTHONPATH && cd src/attacker && python3 ae4dnn.py

open-result_file:
	cat $(PROJECT_DIR)/src/attacker/result/ae4dnn/ae4dnn_targetmodel_all_7weight=0,005_1000.txt
