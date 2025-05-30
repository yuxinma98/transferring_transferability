import os

color_dict = {
    "SVD-DS": "#dc267f",
    "SVD-DS (Normalized)": "#785ef0",
    "DS-CI (Normalized)": "#FE6100",
    "DS-CI (Compatible)": "#648fff",
    # "OI-DS (Normalized)": "#FFB000",
}

plot_model_names = {
    "SVD-DS": "Unnormalized SVD-DS (incompatible)",
    "SVD-DS (Normalized)": "Normalized SVD-DS (transferable)",
    "DS-CI (Normalized)": "Normalized DS-CI \n(approximately transferable)",
    "DS-CI (Compatible)": "Compatible DS-CI \n(transferable)",
    # "OI-DS (Normalized)": "Normalized OI-DS \n(approximately compatible, continuous)",
}

data_dir = f"{os.environ.get('DATA_DIR')}/anydim_transferability/OI-DS/"
