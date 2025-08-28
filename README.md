# Meta-Learning on Protein Mutation Sequence

Welcome to the Meta-learning for Protein Mutate Property Difference Prediction repository! This repository corresponds to the meta-learning approach on Protein Mutation property predict project. 

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kuku0202/maml.git
   cd maml
  
2. ** Create virtual environment to run this project:**
    ```bash
    conda create -n maml python=3.9
    conda activate maml

3. **Install Required Packages:**

Ensure you have Python and pip installed. Then, install the necessary packages:

    pip install -r requirements.txt
    mkdir -p saved_models
    cd saved_models


4. **Download the model:**

To download the model, either:
  ```bash
  git lfs install
  git clone https://huggingface.co/yuesu4/Protein_Mutation_ProtBert_MAML
  ```
  or:
  ```bash
  from huggingface_hub import snapshot_download

  snapshot_download(repo_id="yuesu4/Protein_Mutation_ProtBert_MAML", local_dir="./saved_models/")
  ```
Note: Make sure to install huggingface_hub if using option 2:
  ```bash
  pip install huggingface_hub
  ```
The  model from https://huggingface.co/yuesu4/Protein_Mutation_ProtBert_MAML should be saved in the saved_models directory for use with the pipeline.


## Usage
Run the full pipeline on all datasets:
```bash
python main.py --initial_task preprocess_data/combined_ddG_all.csv 
    --meta_learning_tasks preprocess_data/binding_affinity_train.csv            preprocess_data/ddG/*train.csv preprocess_data/enrichment_score/*train.csv preprocess_data/fireprot_ddG/*train.csv preprocess_data/fireprot_dTm/*train.csv 
    --test_tasks preprocess_data/binding_affinity/*test.csv preprocess_data/ddG/*test.csv preprocess_data/enrichment_score/*test.csv preprocess_data/fireprot_ddG/*test.csv preprocess_data/fireprot_dTm/*test.csv 
    --run_maml 
    --run_finetune 
    --maml_epochs 50 
    --finetune_epochs 5 
    --save_dir ./results_v3