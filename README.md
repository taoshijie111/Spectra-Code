#


# S2M: The Project for Molecule SELFIES predition based on spectrum using BART model.
## Code Structure
    - configs: configuration files
    - data: dataset and dataloader
    - helper: some tools code
    - logs: best model checkpoint
    - models: BART model 
    - main.py: train script

## Training From Scratch
    python main.py --device cuda:0 --model_name bart --dataset qm9 --train ir_tain.mdb --test ir_test.mdb

## Inference and Analysis
    result_analysis.ipynb
