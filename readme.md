# NLP Tests

This repository contains a collection of scripts and configurations for training and testing homegrown NLP models. The project is structured to facilitate experimentation with different configurations and datasets.
Mainly it is so I can play around with different normalisation or magnitude preserving techniques.

Currently of interest are:
 - nGPT: https://arxiv.org/pdf/2410.01131v1.
 - Magnitude Preserving layers: https://arxiv.org/pdf/2312.02696.

<div align: center;">
  <img src="https://github.com/user-attachments/assets/1fac8343-7b3f-4587-aa86-fc448c86fe69" alt="" width="300"/>
  <img src="https://github.com/user-attachments/assets/313bd529-e569-4c96-9b87-8d44f3f0ae7b" alt="" width="300"/>
  <img src="https://github.com/user-attachments/assets/a230ee36-23e3-431d-8552-ec22816190cd" alt="" width="300"/>
  <img src="https://github.com/user-attachments/assets/b59c00ab-e27d-490f-808e-95ac095a746d" alt="" width="300"/>
  <img src="https://github.com/user-attachments/assets/4b4d64f3-8757-441c-adff-ed61ed5477a1" alt="" width="300"/>
</div>

## Project Structure

```
├── configs
│   ├── datamodule
│   ├── experiment
│   │   ├── all_rms.yaml
│   │   ├── all_token.yaml
│   │   └── vanilla.yaml
│   ├── hydra
│   ├── model
│   │   ├── gpt.yaml
│   │   └── mpgpt.yaml
│   └── train.yaml
├── data
│   └── wikitext-103
├── outputs
├── scripts
│   ├── generate.py
│   ├── setup_openwebtext.py
│   ├── setup_wikitext.py
│   └── train.py
├── src
│   ├── datamodules
│   ├── hydra_utils.py
│   ├── layers
│   ├── models
│   │   ├── gpt.py
│   │   └── mpgpt.py
│   ├── schedulers.py
│   └── torch_utils.py
├── LICENSE
├── pyproject.toml
├── readme.md
├── requirements.txt
└── .pre-commit-config.yaml
```

### Key Directories and Files

- **configs/**: Contains configuration files for experiments, Hydra, and training.
- **data/**: Directory for storing datasets.
- **outputs/**: Directory for storing output files and logs.
- **scripts/**: Contains scripts for running training and setup tasks.
- **src/**: Source code for data modules, models, and utilities.

## Usage

Right now the main functionality is just training the autoregressive model on either WikiText-103 or OpenWebText.
mainly for watching the loss go down or using the generate script to see inspect the quality.

### Data Preparation

To prepare the WikiText-103 dataset, run the `setup_wikitext.py` script:

```sh
python scripts/setup_wikitext.py
```

This script downloads, tokenizes, and saves the dataset in the `wikitext-103` directory.
This saves the dataset as numpy arrays which are only a couple of GB in size, and can be loaded wholly into memory.
The same can be done for OpenWebText but this takes quite long and requires ~50GB of HDD space.

### Training

To start training a model, run the `train.py` script located in the `scripts` directory:

```sh
python scripts/train.py
```
### Configuration

The project uses Hydra for managing configurations.
The main configuration file is located at `configs/train.yaml`.

You can override any configuration parameter from the command line.
For example, to change the seed and the precision, you can run:

```sh
python scripts/train.py seed=123 precision=high
```
Alternativel large collections of keys may be be overwritten by files in the `configs/experiment` directory.
Once created, they can be loaded using

```sh
python scripts/train.py experiment=new_file
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
