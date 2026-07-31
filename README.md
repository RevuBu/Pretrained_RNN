# Pretraining of Embodied Recurrent Networks Bridges the Gap Between Artificial and Cortical Neural Activities

This repository accompanies the paper *"Pretraining of Embodied Recurrent Networks Bridges the Gap Between Artificial and Cortical Neural Activities"*. It trains RNNs in a biomechanical simulation environment (MotorNet) through a developmental pretraining pipeline and evaluates alignment between model hidden activity and macaque motor cortex recordings using CCA  and DSA.

## Repository Structure

```
Pretrained_RNN/
├── model/                  # RNNCell (Dale's law, sparse connectivity) & builders
├── tasks/                  # Motor tasks: grid reach, center-out, double reach, RTT
├── training/               # Training scripts & hyperparameters
├── scripts/                # Data preprocessing (firing rates, CCA/AUC)
├── analysis/               # Neural similarity analysis & statistics
├── figures/                # Paper figure generation (Fig 3–7)
├── Data/                   # Neural data & precomputed results
│   ├── CO/                 # Center-out task data
│   ├── DR/                 # Double-reach task data
│   └── RTT/                # Random-target task data
├── Results/                # Output directory
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

**Training** (progressive stages):

```bash
python -m training.pretrain_grid   # Stage 1: grid reach pretraining
python -m training.train_co        # Stage 2: center-out fine-tuning
python -m training.train_dr        # Stage 3: double-reach training
python -m training.train_rtt       # Stage 4: random-target sequence training
```

**Data preprocessing & analysis**:

```bash
python -m scripts.pre_data         # Generate model firing rates
python -m scripts.pre_CO_data      # Compute CCA/AUC
python -m analysis.eval_BasicModel # Behavioral evaluation
python -m analysis.Compute_CO_DSA  # DSA computation
python -m analysis.Melting         # Cross-model CCA analysis
```

**Figures**:

```bash
python -m figures.fig3   # Fig 3–7 generation
```

## Model

RNNCell ([model/rnn.py](file:///e:/Modelling/Pretrained_RNN/model/rnn.py)) features Dale's law (80% excitatory / 20% inhibitory), sparse connectivity (recurrent density 0.8), spectral radius normalization (1.5), a separate task-input MLP, and configurable sensory feedback (proprioceptive, visual, or combined). Time constants: dt=10ms, tau=50ms.

## Data

* **`Data/CO/`**, **`Data/DR/`**, **`Data/RTT/`** — model firing rates, CCA/AUC/DSA results, and trajectory data for each task.

* Monkey C (`fr_MC_*.npy`) and Monkey M (`fr_MM_*.npy`) motor cortex recordings are included for neural comparison.

* Pretrained weights are in `model/`.

