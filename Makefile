##### PATHS #####

VERSION=v1.6
BASE_ENV_VERSION=v1.6
PROJECT_ID=neuro-project-7374990e

DATA_DIR?=data
CONFIG_DIR?=configs
CODE_DIR?=modules
SCRIPT_DIR?=scripts
NOTEBOOKS_DIR?=notebooks
RESULTS_DIR?=results

PROJECT_PATH_STORAGE?=storage:microtubules-classifiaction

PROJECT_PATH_ENV?=/project

##### JOB NAMES #####

PROJECT_POSTFIX?=microtubules-classifiaction

SETUP_JOB?=setup-$(PROJECT_POSTFIX)
TRAIN_JOB?=train-$(PROJECT_POSTFIX)
DEVELOP_JOB?=develop-$(PROJECT_POSTFIX)
TENSORBOARD_JOB?=tensorboard-$(PROJECT_POSTFIX)
FILEBROWSER_JOB?=filebrowser-$(PROJECT_POSTFIX)

##### ENVIRONMENTS #####

BASE_ENV_NAME?=neuromation/base:$(BASE_ENV_VERSION)
CUSTOM_ENV_NAME?=image:neuromation-$(PROJECT_POSTFIX):$(VERSION)

##### VARIABLES YOU MAY WANT TO MODIFY #####

# Jupyter mode. Available options: notebook (to run Jupyter Notebook), lab (to run JupyterLab).
JUPYTER_MODE?=notebook

# Location of your dataset on the platform storage. Example:
# DATA_DIR_STORAGE?=storage:datasets/cifar10
DATA_DIR_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_DIR)

# The type of the training machine (run `neuro config show` to see the list of available types).
PRESET?=gpu-small

# HTTP authentication (via cookies) for the job's HTTP link.
# Applied only to jupyter, tensorboard and filebrowser jobs.
# Set `HTTP_AUTH=--no-http-auth` to disable any authentication.
# WARNING: removing authentication might disclose your sensitive data stored in the job.
HTTP_AUTH?=--http-auth

# When running the training job, wait until it gets actually running,
# and stream logs to the standard output.
# Set any other value to disable this feature: `TRAIN_STREAM_LOGS=no`.
TRAIN_STREAM_LOGS?=yes

# Command to run training inside the environment. Example:
TRAIN_CMD?=python -u $(CODE_DIR)/train.py --config_path $(PROJECT_PATH_ENV)/$(CONFIG_DIR)/config.sample.py

LOCAL_PORT?=1489


##### SECRETS ######

# Google Cloud integration settings:
GCP_SECRET_FILE?=neuro-job-key.json
GCP_SECRET_PATH_LOCAL=$(CONFIG_DIR)/$(GCP_SECRET_FILE)
GCP_SECRET_PATH_ENV=$(PROJECT_PATH_ENV)/$(GCP_SECRET_PATH_LOCAL)

# AWS integration settings:
AWS_SECRET_FILE?=aws-credentials.txt
AWS_SECRET_PATH_LOCAL=$(CONFIG_DIR)/$(AWS_SECRET_FILE)
AWS_SECRET_PATH_ENV=$(PROJECT_PATH_ENV)/$(AWS_SECRET_PATH_LOCAL)

# Weights and Biases integration settings:
WANDB_SECRET_FILE?=wandb-token.txt
WANDB_SECRET_PATH_LOCAL=$(CONFIG_DIR)/$(WANDB_SECRET_FILE)
WANDB_SECRET_PATH_ENV=$(PROJECT_PATH_ENV)/$(WANDB_SECRET_PATH_LOCAL)
WANDB_SWEEP_CONFIG_FILE?=wandb-sweep.yaml
WANDB_SWEEP_CONFIG_PATH=$(CODE_DIR)/$(WANDB_SWEEP_CONFIG_FILE)
WANDB_SWEEPS_FILE=.wandb_sweeps

##### COMMANDS #####

NEURO?=neuro

ifeq ($(TRAIN_STREAM_LOGS), yes)
	TRAIN_WAIT_START_OPTION=--wait-start --detach
else
	TRAIN_WAIT_START_OPTION=
endif

# Check if GCP authentication file exists, then set up variables
ifneq ($(wildcard $(GCP_SECRET_PATH_LOCAL)),)
	OPTION_GCP_CREDENTIALS=\
		--env GOOGLE_APPLICATION_CREDENTIALS="$(GCP_SECRET_PATH_ENV)" \
		--env GCP_SERVICE_ACCOUNT_KEY_PATH="$(GCP_SECRET_PATH_ENV)"
else
	OPTION_GCP_CREDENTIALS=
endif

# Check if AWS authentication file exists, then set up variables
ifneq ($(wildcard $(AWS_SECRET_PATH_LOCAL)),)
	OPTION_AWS_CREDENTIALS=\
		--env AWS_CONFIG_FILE="$(AWS_SECRET_PATH_ENV)" \
		--env NM_AWS_CONFIG_FILE="$(AWS_SECRET_PATH_ENV)"
else
	OPTION_AWS_CREDENTIALS=
endif

# Check if Weights & Biases key file exists, then set up variables
ifneq ($(wildcard $(WANDB_SECRET_PATH_LOCAL)),)
	OPTION_WANDB_CREDENTIALS=--env NM_WANDB_TOKEN_PATH="$(WANDB_SECRET_PATH_ENV)"
else
	OPTION_WANDB_CREDENTIALS=
endif

##### HELP #####

.PHONY: help
help:
	@# generate help message by parsing current Makefile
	@# idea: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -hE '^[a-zA-Z_-]+:[^#]*?### .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##### SETUP #####

.PHONY: setup
setup: ### Setup remote environment
	$(NEURO) mkdir --parents $(PROJECT_PATH_STORAGE) \
		$(PROJECT_PATH_STORAGE)/$(CODE_DIR) \
		$(DATA_DIR_STORAGE) \
		$(PROJECT_PATH_STORAGE)/$(CONFIG_DIR) \
		$(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR) \
		$(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)
	$(NEURO) cp requirements.txt $(PROJECT_PATH_STORAGE)
	$(NEURO) cp apt.txt $(PROJECT_PATH_STORAGE)
	$(NEURO) cp setup.cfg $(PROJECT_PATH_STORAGE)
	$(NEURO) run \
		--name $(SETUP_JOB) \
		--description "$(PROJECT_ID):setup" \
		--preset cpu-small \
		--detach \
		--env JOB_TIMEOUT=1h \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):ro \
		$(BASE_ENV_NAME) \
		'sleep infinity'
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'export DEBIAN_FRONTEND=noninteractive && apt-get -qq update && cat $(PROJECT_PATH_ENV)/apt.txt | tr -d \"\\r\" | xargs -I % apt-get -qq install --no-install-recommends % && apt-get -qq clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*'"
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'pip install --progress-bar=off -U --no-cache-dir -r $(PROJECT_PATH_ENV)/requirements.txt'"
	$(NEURO) --network-timeout 300 job save $(SETUP_JOB) $(CUSTOM_ENV_NAME)
	$(NEURO) kill $(SETUP_JOB) || :
	@touch .setup_done

.PHONY: kill-setup
kill-setup:  ### Terminate the setup job (if it was not killed by `make setup` itself)
	$(NEURO) kill $(SETUP_JOB) || :

.PHONY: _check_setup
_check_setup:
	@test -f .setup_done || { echo "Please run 'make setup' first"; false; }

##### STORAGE #####

.PHONY: upload-code
upload-code: TO_CODE_DIR=$(CODE_DIR)
upload-code: _check_setup  ### Upload code directory to the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(CODE_DIR) $(PROJECT_PATH_STORAGE)/$(TO_CODE_DIR)
	$(NEURO) cp --recursive --update --no-target-directory $(SCRIPT_DIR) $(PROJECT_PATH_STORAGE)/$(SCRIPT_DIR)

.PHONY: download-code
download-code: _check_setup  ### Download code directory from the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(PROJECT_PATH_STORAGE)/$(CODE_DIR) $(CODE_DIR)

.PHONY: clean-code
clean-code: _check_setup  ### Delete code directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(CODE_DIR)/*

.PHONY: upload-data
upload-data: _check_setup  ### Upload data directory to the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(DATA_DIR) $(DATA_DIR_STORAGE)

.PHONY: download-data
download-data: _check_setup  ### Download data directory from the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(DATA_DIR_STORAGE) $(DATA_DIR)

.PHONY: clean-data
clean-data: _check_setup  ### Delete data directory from the platform storage
	$(NEURO) rm --recursive $(DATA_DIR_STORAGE)/*

.PHONY: upload-config
upload-config: TO_CONFIG_DIR=$(CONFIG_DIR)
upload-config: _check_setup  ### Upload config directory to the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(CONFIG_DIR) $(PROJECT_PATH_STORAGE)/$(TO_CONFIG_DIR)

.PHONY: download-config
download-config: _check_setup  ### Download config directory from the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR) $(CONFIG_DIR)

.PHONY: clean-config
clean-config: _check_setup  ### Delete config directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)/*

.PHONY: upload-notebooks
upload-notebooks: _check_setup  ### Upload notebooks directory to the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(NOTEBOOKS_DIR) $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR)

.PHONY: download-notebooks
download-notebooks: _check_setup  ### Download notebooks directory from the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR) $(NOTEBOOKS_DIR)

.PHONY: clean-notebooks
clean-notebooks: _check_setup  ### Delete notebooks directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR)/*

.PHONY: upload-results
upload-results: _check_setup  ### Upload results directory to the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(RESULTS_DIR) $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)

.PHONY: download-results
download-results: _check_setup  ### Download results directory from the platform storage
	$(NEURO) cp --recursive --update --no-target-directory $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR) $(RESULTS_DIR)

.PHONY: clean-results
clean-results: _check_setup  ### Delete results directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)/*

.PHONY: upload-all
upload-all: upload-code upload-data upload-config upload-notebooks upload-results  ### Upload code, data, config, notebooks, and results directories to the platform storage

.PHONY: download-all
download-all: download-code download-data download-config download-notebooks download-results  ### Download code, data, config, notebooks, and results directories from the platform storage

.PHONY: clean-all
clean-all: clean-code clean-data clean-config clean-notebooks clean-results  ### Delete code, data, config, notebooks, and results directories from the platform storage

##### Google Cloud Integration #####

.PHONY: gcloud-check-auth
gcloud-check-auth:  ### Check if the file containing Google Cloud service account key exists
	@echo "Using variable: GCP_SECRET_FILE='$(GCP_SECRET_FILE)'"
	@test "$(OPTION_GCP_CREDENTIALS)" \
		&& echo "Google Cloud will be authenticated via service account key file: '$$PWD/$(GCP_SECRET_PATH_LOCAL)'" \
		|| { echo "ERROR: Not found Google Cloud service account key file: '$$PWD/$(GCP_SECRET_PATH_LOCAL)'"; \
			echo "Please save the key file named GCP_SECRET_FILE='$(GCP_SECRET_FILE)' to './$(CONFIG_DIR)/'"; \
			false; }

##### AWS Integration #####

.PHONY: aws-check-auth
aws-check-auth:  ### Check if the file containing AWS user account credentials exists
	@echo "Using variable: AWS_SECRET_FILE='$(AWS_SECRET_FILE)'"
	@test "$(OPTION_AWS_CREDENTIALS)" \
		&& echo "AWS will be authenticated via user account credentials file: '$$PWD/$(AWS_SECRET_PATH_LOCAL)'" \
		|| { echo "ERROR: Not found AWS user account credentials file: '$$PWD/$(AWS_SECRET_PATH_LOCAL)'"; \
			echo "Please save the key file named AWS_SECRET_FILE='$(AWS_SECRET_FILE)' to './$(CONFIG_DIR)/'"; \
			false; }

##### WandB Integration #####

.PHONY: wandb-check-auth
wandb-check-auth:  ### Check if the file Weights and Biases authentication file exists
	@echo Using variable: WANDB_SECRET_FILE='$(WANDB_SECRET_FILE)'
	@test "$(OPTION_WANDB_CREDENTIALS)" \
		&& echo "Weights & Biases will be authenticated via key file: '$$PWD/$(WANDB_SECRET_PATH_LOCAL)'" \
		|| { echo "ERROR: Not found Weights & Biases key file: '$$PWD/$(WANDB_SECRET_PATH_LOCAL)'"; \
			echo "Please save the key file named WANDB_SECRET_FILE='$(WANDB_SECRET_FILE)' to './$(CONFIG_DIR)/'"; \
			false; }

##### JOBS #####

.PHONY: develop
develop: _check_setup upload-code upload-config upload-notebooks  ### Run a development job
	$(NEURO) run \
		--name $(DEVELOP_JOB) \
		--description "$(PROJECT_ID):develop" \
		--preset $(PRESET) \
		--http 8888 \
		$(HTTP_AUTH) \
		--browse \
		--detach \
		--volume $(DATA_DIR_STORAGE):$(PROJECT_PATH_ENV)/$(DATA_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(SCRIPT_DIR):$(PROJECT_PATH_ENV)/$(SCRIPT_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR):$(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR):rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
		--env EXPOSE_SSH=yes \
		--env JOB_TIMEOUT=1d \
		$(OPTION_GCP_CREDENTIALS) $(OPTION_AWS_CREDENTIALS) $(OPTION_WANDB_CREDENTIALS) \
		$(CUSTOM_ENV_NAME) \
		jupyter $(JUPYTER_MODE) --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=$(PROJECT_PATH_ENV)

.PHONY: connect-develop
connect-develop:  ### Connect to the remote shell running on the development job
	$(NEURO) exec --no-key-check $(DEVELOP_JOB) bash

.PHONY: logs-develop
logs-develop:  ### Connect to the remote shell running on the development job
	$(NEURO) logs $(DEVELOP_JOB)

.PHONY: port-forward-develop
port-forward-develop:  ### Forward SSH port to localhost for remote debugging
	@test $(LOCAL_PORT) || { echo 'Please set up env var LOCAL_PORT'; false; }
	$(NEURO) port-forward $(DEVELOP_JOB) $(LOCAL_PORT):22

.PHONY: kill-develop
kill-develop:  ### Terminate the development job
	$(NEURO) kill $(DEVELOP_JOB) || :

RUN?=base

.PHONY: train
train: CONFIG=remote_config_sample.yaml
train: ENTRYPOINT=train.py
train: CODE_FROM_DIR=$(PROJECT_PATH_STORAGE)/$(CODE_DIR)
train: CONFIG_FROM_DIR=$(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)
train: TRAIN_CMD=python -u $(CODE_DIR)/$(ENTRYPOINT) --config $(CONFIG)
train: _check_setup upload-code upload-config   ### Run a training job (set up env var 'RUN' to specify the training job),
	$(NEURO) run \
		--name $(TRAIN_JOB)-$(RUN) \
		--description "$(PROJECT_ID):train" \
		--preset gpu-large \
		--wait-start \
		$(TRAIN_WAIT_START_OPTION) \
		--volume $(DATA_DIR_STORAGE):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
		--volume $(CODE_FROM_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):rw \
		--volume $(CONFIG_FROM_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
		--env EXPOSE_SSH=yes \
		--env JOB_TIMEOUT=0 \
		$(OPTION_GCP_CREDENTIALS) $(OPTION_AWS_CREDENTIALS) $(OPTION_WANDB_CREDENTIALS) \
		$(CUSTOM_ENV_NAME) \
		bash -c 'cd $(PROJECT_PATH_ENV) && $(TRAIN_CMD)'
ifeq ($(TRAIN_STREAM_LOGS), yes)
	@echo "Streaming logs of the job $(TRAIN_JOB)-$(RUN)"
	$(NEURO) exec --no-key-check -T $(TRAIN_JOB)-$(RUN) "tail -f /output" || echo -e "Stopped streaming logs.\nUse 'neuro logs <job>' to see full logs."
endif

.PHONY: train-clean
train-clean: SEED:=$(shell bash -c 'echo $$RANDOM')
train-clean:
	( make upload-config TO_CONFIG_DIR=$(CONFIG_DIR)-$(SEED); \
	make upload-code TO_CODE_DIR=$(CODE_DIR)-$(SEED); \
	make train CODE_FROM_DIR=$(PROJECT_PATH_STORAGE)/$(CODE_DIR)-$(SEED) CONFIG_FROM_DIR=$(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)-$(SEED); \
	$(NEURO) rm -r $(PROJECT_PATH_STORAGE)/$(CODE_DIR)-$(SEED); \
	$(NEURO) rm -r $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)-$(SEED) )

.PHONY: kill-train
kill-train:  ### Terminate the training job (set up env var 'RUN' to specify the training job)
	$(NEURO) kill $(TRAIN_JOB)-$(RUN) || :

.PHONY: kill-train-all
kill-train-all:  ### Terminate all training jobs you have submitted
	jobs=`neuro --quiet ps --description="$(PROJECT_ID):train" | tr -d "\r"` && \
	$(NEURO) kill $${jobs:-placeholder} || :

.PHONY: connect-train
connect-train: _check_setup  ### Connect to the remote shell running on the training job (set up env var 'RUN' to specify the training job)
	$(NEURO) exec --no-key-check $(TRAIN_JOB)-$(RUN) bash

.PHONY: jupyter
jupyter:
	$(NEURO) job browse $(DEVELOP_JOB) || :

.PHONY: tensorboard
tensorboard: _check_setup  ### Run a job with TensorBoard and open UI in the default browser
	$(NEURO) run \
		--name $(TENSORBOARD_JOB) \
		--preset cpu-large \
		--description "$(PROJECT_ID):tensorboard" \
		--http 6006 \
		$(HTTP_AUTH) \
		--browse \
		--env JOB_TIMEOUT=1d \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):ro \
		$(CUSTOM_ENV_NAME) \
		bash -c "pip install -U tensorboard && tensorboard --host=0.0.0.0 --logdir=$(PROJECT_PATH_ENV)/$(RESULTS_DIR)"

.PHONY: kill-tensorboard
kill-tensorboard:  ### Terminate the job with TensorBoard
	$(NEURO) kill $(TENSORBOARD_JOB) || :

.PHONY: filebrowser
filebrowser: _check_setup  ### Run a job with File Browser and open UI in the default browser
	$(NEURO) run \
		--name $(FILEBROWSER_JOB) \
		--description "$(PROJECT_ID):filebrowser" \
		--preset cpu-small \
		--http 80 \
		$(HTTP_AUTH) \
		--browse \
		--env JOB_TIMEOUT=1d \
		--volume $(PROJECT_PATH_STORAGE):/srv:rw \
		filebrowser/filebrowser \
		--noauth

.PHONY: kill-filebrowser
kill-filebrowser:  ### Terminate the job with File Browser
	$(NEURO) kill $(FILEBROWSER_JOB) || :

.PHONY: kill-all
kill-all: kill-develop kill-train-all kill-tensorboard kill-filebrowser kill-setup  ### Terminate all jobs of this project

##### LOCAL #####

.PHONY: setup-local
setup-local:  ### Install pip requirements locally
	pip install -r requirements.txt

.PHONY: format
format:  ### Automatically format the code
	isort -rc modules
	black modules

.PHONY: lint
lint:  ### Run static code analysis locally
	isort -c -rc modules
	black --check modules
	mypy modules
	flake8 modules

##### MISC #####

.PHONY: ps
ps:  ### List all running and pending jobs
	$(NEURO) ps

.PHONY: _upgrade
_upgrade:
	@if ! (git status | grep "nothing to commit"); then echo "Please commit or stash changes before upgrade."; exit 1; fi
	@echo "Applying the latest Neuro Project Template to this project..."
	cookiecutter \
		--output-dir .. \
		--no-input \
		--overwrite-if-exists \
		--checkout release \
		gh:neuromation/cookiecutter-neuro-project \
		project_slug=$(PROJECT_POSTFIX) \
		code_directory=$(CODE_DIR)
	git checkout -- $(DATA_DIR) $(CODE_DIR) $(CONFIG_DIR) $(NOTEBOOKS_DIR) $(RESULTS_DIR)
	git checkout -- .gitignore requirements.txt apt.txt setup.cfg README.md
	@echo "Some files are successfully changed. Please review the changes using git diff."

.PHONY: host_data
host_data:
	neuro kill host-data
	neuro run -s cpu-small \
	    -v storage:microtubules-classifiaction/data/to_deploy:/data:rw \
        --http 8686 \
        --no-http-auth \
        --name host-data \
        --life-span=0 \
        --browse \
        --env JOB_TIMEOUT=0 \
        frolvlad/alpine-python3 \
        "python -m http.server 8686"

