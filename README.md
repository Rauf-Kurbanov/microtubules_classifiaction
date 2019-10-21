# microtubules_classifiaction

# Description

Created by Rauf Kurbanov (kurbanov.re@gmail.com)

# Development Environment

This project is designed to run on [Neuro Platform](https://neu.ro), so you can jump into problem-solving right away.

## Directory Structure

| Mount Point                                  | Description           | Storage URI                                                                  |
|:-------------------------------------------- |:--------------------- |:---------------------------------------------------------------------------- |
|`microtubules_classifiaction/data/`                              | Data                  | `storage:microtubules_classifiaction/data/`                              |
|`microtubules_classifiaction/modules/` | Python modules        | `storage:microtubules_classifiaction/modules/` |
|`microtubules_classifiaction/notebooks/`                         | Jupyter notebooks     | `storage:microtubules_classifiaction/notebooks/`                         |
|`microtubules_classifiaction/results/`                           | Logs and results      | `storage:microtubules_classifiaction/results/`                           |

## Development

Follow the instructions below in order to setup the environment and start Jupyter development session.

## Neuro Platform

* Setup development environment `make setup`
* Run Jupyter with GPU: `make jupyter`
* Kill Jupyter: `make kill-jupyter`
* Get the list of available template commands: `make help`

# Data

## Uploading via Web UI

On local machine run `make filebrowser` and open job's URL on your mobile device or desktop.
Through a simple file explorer interface you can upload test images and perform file operations.

## Uploading via CLI

On local machine run `make upload-data`. This command pushes local files stored in `./data`
into `storage:microtubules_classifiaction/data` mounted to your development environment's `/project/data`.
