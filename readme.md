# Magnitude Preserving Transformer

This repository contains a collection of scripts and configurations for training and testing homegrown NLP models.
The project is structured to facilitate experimentation with different configurations and datasets.
Mainly it is so I can play around with different normalisation or magnitude preserving techniques.
At the moment I am interested in a transformer that preserves the magnitude of all outputs in each layer.
Relevant papers:
 - nGPT: https://arxiv.org/pdf/2410.01131v1.
 - Magnitude Preserving layers (EDM2): https://arxiv.org/pdf/2312.02696.
I have ended up somewhere in the middle between these two approaches

## Main ideas
- Make all linear layers magnitude preserving
 - Manually unit_norm weights after each SGD step
 - Manyallu unit_norm weights during foward pass (ensures gradients are orthogonal)
- Make the embedding magnitude preserving in the same way
- Remove all learnable affine transformations from rms norm
- Remove weight decay and learning rate warmup
- Use learnable magnitude preserving residual connections of the form:
$$x \leftarrow \frac{x + \alpha(F(RMSNorm(x) - x))}{\sqrt{(1-\alpha)^2 + \alpha^2}}$$
  - Where F is either MP-SelfAttention or MP-SwiGLU and $\alpha$ is a learnable tensor akin to LayerScale


## Initial Results
- Pre-Norm transformers require strong weight decay (0.1) to ensure signal magnitude remains stable with each layer
- MP layers help the transformer to learn more efficiently without any weight decay
- My limited tests (small model and small batchsize) show MP-GPT improving over standard GPT (PreNorm+QKNorm+LayerScale)
- I don't have access to the required compute for a proper GPT2 scale the testing :(

<img src="https://github.com/user-attachments/assets/dc96a32e-6dce-4abb-ba10-c4d8c84ab014" alt="" width="300"/>
<img src="https://github.com/user-attachments/assets/6631c3a2-1b84-4fde-8630-eca95e8bf3ac" alt="" width="300"/>
<img src="https://github.com/user-attachments/assets/4d1c7553-50b8-4152-b87b-b8e30fe69058" alt="" width="300"/>
<img src="https://github.com/user-attachments/assets/58ebe7bd-a962-4108-a1c6-4d0fbe9a6abd" alt="" width="300"/>
<img src="https://github.com/user-attachments/assets/4e1381ff-20c1-46af-bc9d-324a973eb924" alt="" width="300"/>
<img src="https://github.com/user-attachments/assets/9b896926-f122-41d1-a7e0-0ff43ea4048b" alt="" width="300"/>



## Project Structure

```
├── configs
│   ├── datamodule
│   │   ├── openwebtext.yaml
│   │   └── wikitext.yaml
│   ├── experiment
│   │   ├── all_rms.yaml
│   │   ├── all_token.yaml
│   │   └── vanilla.yaml
│   ├── hydra
│   │   └── default.yaml
│   ├── model
│   │   ├── gpt.yaml
│   │   ├── mpgpt.yaml
│   │   └── ngpt.yaml
│   └── train.yaml
├── data
├── LICENSE
├── outputs
├── pyproject.toml
├── readme.md
├── requirements.txt
├── scripts
│   ├── generate.py
│   ├── setup_openwebtext.py
│   ├── setup_wikitext.py
│   └── train.py
└── src
    ├── datamodules
    │   └── text.py
    ├── hydra_utils.py
    ├── layers
    │   ├── magnitude.py
    │   ├── normalisation.py
    │   └── transformer.py
    ├── models
    │   ├── gpt.py
    │   ├── mpgpt.py
    │   └── ngpt.py
    ├── schedulers.py
    └── torch_utils.py
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
