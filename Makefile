DEV_JOB_NAME ?= tubles-dev
TRAIN_JOB_NAME ?= tubles-train
LOCAL_PORT ?= 1489
CODE_FOLDER ?= storage:tubles/src
DEV_CODE_TAG ?= dev
REPO_NAME ?= text_summarization
TENSORBOARD_NAME ?= tubles-tb
STORAGE_ROOT ?= storage:tubles
TB_FOLDER ?= $(STORAGE_ROOT)/data/logs
IMAGE_NAME ?= text_summarization

.PHONY: setup
setup: ### Setup remote environment
	$(NEURO) mkdir --parents $(STORAGE_ROOT) \
		$(PROJECT_PATH_STORAGE)/$(CODE_DIR) \
		$(PROJECT_PATH_STORAGE)/data \
		$(PROJECT_PATH_STORAGE)/configs \
		$(PROJECT_PATH_STORAGE)/notebooks \
		$(RESULTS_DIR_STORAGE)
	$(NEURO) run \
		--name $(SETUP_JOB) \
		--description "$(PROJECT_ID):setup" \
		--preset cpu-small \
		--detach \
		--env JOB_TIMEOUT=1h \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):ro \
		$(BASE_ENV_NAME) \
		'sleep infinity'
	for file in $(PROJECT_FILES); do $(NEURO) cp ./$$file $(PROJECT_PATH_STORAGE)/$$file; done
	$(NEURO) exec --no-key-check $(SETUP_JOB) "bash -c 'export DEBIAN_FRONTEND=noninteractive && $(APT) update && cat $(PROJECT_PATH_ENV)/apt.txt | xargs -I % $(APT) install --no-install-recommends % && $(APT) clean && $(APT) autoremove && rm -rf /var/lib/apt/lists/*'"
	$(NEURO) exec --no-key-check $(SETUP_JOB) "bash -c '$(PIP) -r $(PROJECT_PATH_ENV)/requirements.txt'"
	$(NEURO) exec --no-key-check $(SETUP_JOB) "sed -e 's/: True/: False/' -i /usr/local/lib/python3.6/dist-packages/streamlit/server/Server.py"
	$(NEURO) --network-timeout 300 job save $(SETUP_JOB) $(CUSTOM_ENV_NAME)
	$(NEURO) kill $(SETUP_JOB) || :
	@touch .setup_done

.PHONY: dev
dev:
	neuro run -s gpu-small \
	--http 8888 \
	--no-http-auth \
	--volume $(STORAGE_ROOT)/data:/data:rw \
	--volume $(CODE_FOLDER)/$(DEV_CODE_TAG)/$(REPO_NAME):/app:rw \
    --volume $(TB_FOLDER):/logs:rw \
	--name $(DEV_JOB_NAME) \
	--detach \
	image:$(IMAGE_NAME)

.PHONY: kill_dev
kill_dev:
	neuro kill $(DEV_JOB_NAME)


.PHONY: tensorboard
tensorboard:
	neuro run -s cpu-small \
	--volume $(TB_FOLDER):/result:ro \
	--http 6006 \
	--no-http-auth \
	-e GCS_READ_CACHE_MAX_SIZE_MB=0 \
	--name $(TENSORBOARD_NAME) \
	--browse \
	--detach \
	tensorflow/tensorflow \
	"tensorboard --host=0.0.0.0 --logdir=/result"

.PHONY: kill_tensorboard
kill_tensorboard:
	neuro kill $(TENSORBOARD_NAME)

.PHONY: browse_tensorboard
browse_tensorboard:
	neuro job browse $(TENSORBOARD_NAME)


.PHONY: connect
connect:
	neuro exec -t $(DEV_JOB_NAME) bash


.PHONY: browse
browse:
	neuro job browse $(DEV_JOB_NAME)


.PHONY: jupyter
jupyter:
	( make dev; make browse)


.PHONY: port_local
port_local:
	neuro port-forward $(DEV_JOB_NAME) $(LOCAL_PORT):22


.PHONY update_code:
update_code:
	neuro cp -r . $(CODE_FOLDER)/$(DEV_CODE_TAG)


.PHONY: filebrowser
filebrowser: FILEBROWSER_NAME=inga-filebrowser
filebrowser:  ### Run a job with File Browser and open UI in the default browser
	neuro run \
		--name $(FILEBROWSER_NAME) \
		--preset cpu-small \
		--http 80 \
		--no-http-auth \
		$(HTTP_AUTH) \
		--browse \
		--detach \
		--volume $(STORAGE_ROOT):/srv:rw \
		--env PLATFORMAPI_SERVICE_HOST="." \
		filebrowser/filebrowser

.PHONY: host_data
host_data:
	neuro kill host-data
	neuro run -s cpu-small \
	    -v storage:microtubules_classifiaction/data/to_deploy:/data:rw \
        --http 8686 \
        --no-http-auth \
        --name host-data \
        --browse \
        frolvlad/alpine-python3 \
        "python -m http.server 8686"
