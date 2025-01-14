# NLP Tests

This repository contains a collection of scripts and configurations for training and testing homegrown NLP models. The project is structured to facilitate experimentation with different configurations and datasets.
Mainly it is so I can play around with different normalisation techniques...

## Project Structure

```
├── configs
│   ├── experiment
│   │   ├── all_rms.yaml
│   │   ├── all_token.yaml
│   │   └── vanilla.yaml
│   ├── hydra
│   │   └── default.yaml
│   └── train.yaml
├── data
│   └── wikitext-103
├── pyproject.toml
├── readme.md
├── requirements.txt
├── scripts
│   ├── run.sh
│   ├── setup_wikitext.py
│   └── train.py
└── src
    ├── datamodules
    │   └── text.py
    ├── hydra_utils.py
    ├── layers
    │   ├── normalisation.py
    │   └── transformer.py
    ├── models
    │   └── gpt.py
    └── schedulers.py
```

### Key Directories and Files

- **configs/**: Contains configuration files for experiments, Hydra, and training.
- **data/**: Directory for storing datasets.
- **scripts/**: Contains scripts for running training and setup tasks.
- **src/**: Source code for data modules, models, and utilities.
- **requirements.txt**: Lists the dependencies required for the project.
- **pyproject.toml**: Configuration file for project metadata and dependencies.

## Usage

Right now the main functionality is just training the autoregressive model on the WikiText-103 dataset and watching the loss go down.

### Data Preparation

To prepare the WikiText-103 dataset, run the `setup_wikitext.py` script:

```sh
python scripts/setup_wikitext.py
```

This script downloads, tokenizes, and saves the dataset in the `wikitext-103` directory.
This saves the dataset as numpy arrays which are only a couple of GB in size, and can be loaded wholly into memory.

### Training

To start training a model, run the `train.py` script located in the `scripts` directory:

```sh
python scripts/train.py
```

This script uses Hydra for configuration management. You can override any configuration parameter from the command line. For example, to change the seed and the precision, you can run:

```sh
python scripts/train.py seed=123 precision=high
```

### Configuration

The project uses Hydra for managing configurations. The main configuration file is located at `configs/train.yaml`. You can modify this file to change the default settings for your experiments.

Example configuration (`configs/train.yaml`):

```yaml
# @package _global_

defaults:
  - _self_
  - hydra: default.yaml
  - experiment: null

seed: 42
project_name: gpt
network_name: test
output_dir: /srv/beegfs/scratch/groups/rodem/nlp/
ckpt_path: null
weight_ckpt_path: null
precision: medium
compile: null
tags: null
full_resume: False
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
