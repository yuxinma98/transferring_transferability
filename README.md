# On Transferring Transferability
This repository is the official implementation of **On Transferring Transferability: Towards a Theory for Size Generalization**. 

## Requirements

* **Python:** Python 3.9
* **Dependencies:** See `environment.yml` for the full list of required packages.
If you have Anaconda or Miniconda installed, you can run the following command to  create a new Conda environment named `transferability`  and install all required packages. You may need to adjust the `environment.yml` file if you do not have a GPU with CUDA support.
    ```bash
    conda env create -f environment.yml -n transferability
    ```

## Run the Transferability Experiments

The transferability experiments, which evaluate the outputs of untrained, randomly initialized models, are conducted in Jupyter notebooks.

1. **Neural Networks on Sets (DeepSet, Normalized DeepSet, PointNet)**  
   Run the notebook located at:  `DeepSet/transferability.ipynb`  to generate Figure 1 in the paper.

2. **Graph Neural Networks (MPNN, IGN, GGNN [our proposal], Continuous GGNN [our proposal])**  
   Run the notebook located at:  `GNN/transferability.ipynb` to generate Figure 2 in the paper.

3. **Invariant Networks on Point Clouds (DS-CI, SVD-DS [our proposal])**  
   Run the notebook located at:  `O_d/transferability.ipynb` to generate Figure 4 in the paper.

## Run the Size Generalization Experiments

### Size Generalization on Sets

1. **Experiment 1: Population Statistics**
   * To generate the data, run the following command:
     ```bash
     bash DeepSet/data_generator/generate.sh
     ```
     The code for data generation (`DeepSet/data_generator/`) is a modification of the original repository by [manzilzaheer](https://github.com/manzilzaheer/DeepSets), which is associated with the DeepSets paper [1].

   * To rerun the size generalization experiment, first delete the corresponding log files in `DeepSet/log/size_generalization`, then run:
     ```bash
     python -m Anydim_transferability.DeepSet.size_generalization_popstats
     ```
     If you do not remove the log files, the script will instead generate plots based on the existing logs.

2. **Experiment 2: Maximal Distance from the Origin**
   * To rerun the size generalization experiment, delete the relevant log files in `DeepSet/log/size_generalization`, then run:
     ```bash
     python -m Anydim_transferability.DeepSet.size_generalization_popstats
     ```
     This includes the data generation procedure.

### **Size generalization on graphs**
* To rerun the size generalization experiment on graph generation model `[model]` (chosen from `full_random` or `SBM`), first delete the corresponding log files in `GNN/log/size_generalization`, then run:
     ```bash
     python -m Anydim_transferability.GNN.size_generalization --graph_model [model]
     ```
   The implementation of IGN (`GNN/ign_layers.py`) is taken from the original repository of [HyTruongSon](https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch), which is an unofficial implementation of the IGN paper [2].
### **Size generalization on point clouds**
* To generate the data, run the following command:
   ```bash
   python -m Anydim_transferability.O_d.data_generator
   ```
   The code for data generation is partially adapted from the original repository by [nhuang37](https://github.com/nhuang37/InvariantFeatures), which is associated with the paper [3].
* To rerun the size generalization experiment, first delete the corresponding log files in `O_d/log/size_generalization`, then run:
   ```bash
   python -m Anydim_transferability.O_d.size_generalization
   ```
   The implementation of DS-CI and OI-DS is also adapted from [nhuang37](https://github.com/nhuang37/InvariantFeatures/).

## Reference
[1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R.R. and Smola, A.J., 2017. Deep sets. Advances in neural information processing systems, 30.

[2] Maron, H., Ben-Hamu, H., Shamir, N. and Lipman, Y., 2018. Invariant and equivariant graph networks. arXiv preprint arXiv:1812.09902.

[3] Blum-Smith, B., Huang, N., Cuturi, M. and Villar, S., 2024. Learning functions on symmetric matrices and point clouds via lightweight invariant features. CoRR.
