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

## Run the size generalization experiments
### Size generalization on sets
1. **Experiment 1: Population statistics**
   * To generate data, run command
      ```bash
      bash DeepSet/data_generator/generate.sh
      ```
      The code for data generation (`DeepSet/data_generator/`)is a modification of the original repository by [manzilzaheer]("https://github.com/manzilzaheer/DeepSets), which is associated with the DeepSet paper [1].
   * To run the size generalization experiment, run command
      ```bash
      python -m Anydim_transferability.DeepSet.size_generalization_popstats
      ```

2. **Experiment 2: Maximal Distance from the Origin.**
   * To run the size generalization experiment, run command
      ```bash
      python -m Anydim_transferability.DeepSet.size_generalization_popstats
      ```
### **Size generalization on graphs**
### **Size generalization on point clouds**


## Reference
[1] Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems 30 (2017).