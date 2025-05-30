import os

color_dict = {
    "DeepSet": "#FFB000",
    "Normalized DeepSet": "#785ef0",
    "PointNet": "#FE6100",
}

plot_model_names = {
    "DeepSet": "DeepSet (incompatible)",
    "Normalized DeepSet": "Normalized DeepSet (transferable)",
    "PointNet": "PointNet (compatible, not transferable)",
}

plot_model_names_hausdorff = {
    "DeepSet": "DeepSet (incompatible)",
    "Normalized DeepSet": "Normalized DeepSet \n(compatible, not transferable)",
    "PointNet": "PointNet (transferable)",
}

data_dir = f"{os.environ.get('DATA_DIR')}/anydim_transferability/deepset/"
