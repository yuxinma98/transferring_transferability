import os

color_dict = {
    "GNN": "#dc267f",
    "IGN": "#fe6100",
    # "Pointwise-compatible-unreduced": "#ffb000",
    "GGNN": "#FFB000",
    "Continuous GGNN": "#785ef0",
}

plot_model_names = {
    "GNN": "GNN (transferable)",
    "IGN": "Normalized 2-IGN (incompatible)",
    "GGNN": "GGNN (compatible, not transferable)",
    "Continuous GGNN": "Continuous GGNN (transferable)",
}

data_dir = f"{os.environ.get('DATA_DIR')}/anydim_transferability/GNN/"
