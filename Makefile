.PHONY: help prepare-dev test lint run doc

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3

PROJECT_DIR = $(shell pwd)
PROJECT_NAME = AdvGeneration

run-ae4dnn:
	$(VENV_ACTIVATE) && export PYTHONPATH=$(PROJECT_DIR)/src:$PYTHONPATH && cd src/attacker && python3 ae4dnn.py

run-aae:
	$(VENV_ACTIVATE) && export PYTHONPATH=$(PROJECT_DIR)/src:$PYTHONPATH && cd src/attacker && python3 aae_rerank.py
run-fgsm:
	$(VENV_ACTIVATE) && export PYTHONPATH=$(PROJECT_DIR)/src:$PYTHONPATH && cd src/attacker && python3 fgsm.py
run-lbfgs:
	$(VENV_ACTIVATE) && export PYTHONPATH=$(PROJECT_DIR)/src:$PYTHONPATH && cd src/attacker && python3 l_bfgs.py
open-result_file:
	cat $(PROJECT_DIR)/src/attacker/result/ae4dnn/ae4dnn_targetmodel_all_7weight=0,005_1000.txt
