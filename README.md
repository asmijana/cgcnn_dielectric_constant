# CGCNN for Electronic dielectric constant prediction from Matminer data

This repository contains a complete machine learning pipeline for predicting the **Electronic dielectric constant** of materials using atomic features.

## Summary

- Dataset: 1056 materials with Electronic dielectric constant data.
- Features: Using CGCNN
- Targets: Electronic dielectric constant
- Models: CGCNN

## Key Results

**Train  |  Validation  |  Test  |**  
MAE   |  1.238  |  1.066  |  1.061  |  
RMSE  |  7.681  |  3.228  |  2.352  |  
R$^2$ |  0.760  |  0.767  |  0.833  | 

## 📁 Project Structure

```
cgcnn_bandgap/
├── data/
│   ├── processed/        
│       ├── cif/                       # Contains all the cif files
│           ├── id_prop.csv            # Contains id, target for run
│           ├── all_prop.csv           # Contains id, target for all data
│           ├── train_prop.csv         # Contains id, target for training data
│           ├── val_prop.csv           # Contains id, target for validation data
│           ├── train_val_prop.csv     # Contains id, target for training+validation data
│           ├── test_prop.csv          # Contains id, target for test data
│           ├── atom_init.json         # element embeddings for CGCNN
│           ├── mp_000000.cif          # cif for structure 0
│           ├── ...
│           ├── ...
│           ├── mp_xxxxxx.cif          # cif for structure 1055
├── external/
│   ├── cgcnn/                         # CGCNN code
│       ├── main.py                    # Modified main.py code
│       ├── predict.py                 # Modified predict.py code
│       ├── backup_main.py             # Original main.py code
│       ├── backup_predict.py          # Original predict.py code
│       ├── cgcnn/
│           ├── data.py                # Modified data.py code                     
├── model/
│   ├── checkpoint.pth.tar             # Generated after training
│   ├── model_best.pth.tar             # Generated after training
│   ├── train_val_loss.csv             # Generated after training; contains loss data
├── results/
│   ├── elec_parity.pdf                # Parity plot for electronic dielectric constant
│   ├── loss_fn.pdf                    # Plot for train and val loss as a function of epochs
│   ├── metrics.csv                    # Contains MAE, RMSE, and R^2 of train, val, and test datasets
│   ├── train_results.csv              # Contains id, true target, and predicted target values for train dataset
│   ├── val_results.csv                # Contains id, true target, and predicted target values for val dataset
│   ├── test_results.csv               # Contains id, true target, and predicted target values for test dataset 
├── dielectric_constant_cgcnn.ipynb    # Python notebook to run
├── environment.yml                    # Environment dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## How to Run

After cloning, install dependencies:

```bash
conda env create -f environment.yml
conda activate cgcnn-env
```

Open the python notebook and run all cells

Open results/ to obtain metrics.csv, loss_fn.pdf, and parity_plot.pdf


## Author

Created by Asmita Jana (asmitajana[at]gmail[dot]com)  
This project was built as a scientific benchmark for CGCNN on electronic dielectric constant datasets.  
Please cite Xie and Grossman (2018): Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical review letters, 120(14), 145301.  

---

Feel free to use this as a template or baseline for your own ML materials science projects!
