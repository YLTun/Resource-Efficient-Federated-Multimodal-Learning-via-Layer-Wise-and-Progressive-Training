# Resource-Efficient-Federated-Multimodal-Learning-via-Layer-Wise-and-Progressive-Training

This repository contains the simulation code for **LW-FedMML** (Layer-Wise Federated Multi-Modal Learning) and **Prog-FedMML** (Progressive Federated Multi-Modal Learning).

## 🛠️ Environment Setup

You can set up the environment using Anaconda. The dependencies are listed in the provided YAML file: `torch_multimodal_env.yml`

> **Note:** Strict adherence to the versions in the YAML file is not required. You may use your own environment and install specific packages as necessary.

## 📂 Data Preparation

Please download the **COCO** and **ADVANCE** datasets manually before running the experiments.

### 1. Download Pre-trained Weights
Before running any training scripts, you must download/save the required pre-trained weights.
*   Run the notebook: `layer_weights_saver.ipynb`

### 2. Configuration
All configuration files are located in the `/configs/` directory.
> **Important:** Before running the notebooks, open the corresponding config files and **update the dataset directory paths** to match your local machine.

---

## 🚀 Running Experiments

### Option A: COCO Dataset

For the COCO dataset, you can proceed directly to training (ensure config paths are set).

*   **For LW-FedMML:** Run `LW_SUP_ADVANCE_FL.ipynb`
*   **For Prog-FedMML:** Run `PROG_SUP_ADVANCE_FL.ipynb`

---

### Option B: ADVANCE Dataset

For the ADVANCE dataset, you must generate the data splits and client distributions before training.

**Step 1: Data Pre-processing**
1.  Run `ADVANCE_train_test-split.ipynb` to generate train and test splits.
2.  Run `ADVANCE_dirichlet_distribution.ipynb` to generate Federated Learning client splits.

**Step 2: Training**
*   **For LW-FedMML:** Run `LW_SUP_ADVANCE_FL.ipynb`
*   **For Prog-FedMML:** Run `PROG_SUP_ADVANCE_FL.ipynb`

```
