"""Useful for plotting purposes"""
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import pandas as pd

def save_of_strain_traction(
    images: List[np.ndarray],
    displacements:List[np.ndarray],
    results:Dict,
    save_path:Path,
    pixel_size:float,
    implot: int,
    vmaxstrain: float,
    scale: float,
    step: int,
    threshold_inf: float,
    threshold_sup: float,
    show=False):
    """
    Saved a .png containing 
        - The GT vertical displacement and the optical-flow based vertical displacement for the selected image implot
        - The GT strain and the optical-flow base computed strain for the selected image implot
        - The GT stress and the optical-flow base computed stress for the selected image implot

    Args:
        displacements (List[np.ndarray]): List of ground-truth displacement fields
        images (List[np.ndarray]): List of grayscale images corresponding to each displacement field.
        results (Dict): Results dictionary from `compute_of_strain_traction` containing GT and 
            optical-flow-based data (flows, strain, stress, etc.)
        save_path (Path): Path to the directory or filename where the .png figure will be saved.
        pixel_size (float): Conversion factor from pixel to physical units (e.g., µm/pixel).
        implot (int): Index of the image to plot from the input lists.
        vmaxstrain (float): Maximum strain value for color normalization in plots.
        scale (float): Scaling factor for quiver or vector field visualization.
        step (int): Sampling step for displaying displacement vectors (e.g., 1 = every pixel, 2 = every second pixel).
        threshold_inf (float): Lower intensity threshold for image display.
        threshold_sup (float): Upper intensity threshold for image display.
        show (bool, optional): If True, displays the generated figure interactively in addition to saving it.
            Defaults to False.
    """
    
    if not isinstance(list(results.keys())[0], int):
        results = {0: results}

    all_methods = list(results[implot]["flows"].keys())
    methods = [m for m in all_methods if m != "gt"]

    name_map = {
        "GT": "Ground Truth",
        "fista": "Proposed",
        "hs": "HS",
        "ilk": "ILK",
        "tv_l1": "TV-L1",
        "farneback": "Farneback",
    }

    method_names = []
    for m in methods:
        name = m.__name__.replace("_of", "") if callable(m) else str(m)
        method_names.append(name_map.get(name, name)) 

    num_methods = len(methods) + 1  # +1 for Ground Truth

    fig = plt.figure(figsize=(4*num_methods, 12))
    gs = gridspec.GridSpec(3, num_methods, figure=fig,
                           wspace=0.05, hspace=0.05)

    gt = displacements[implot][0, 0] * pixel_size
    data_list_of = [gt] + [
        results[implot]["flows"][m][0, 0] for m in methods
    ]

    vmin = min(np.min(d) for d in data_list_of)
    vmax = max(np.max(d) for d in data_list_of)

    ims = []
    axes_line = []

    for j, data in enumerate(data_list_of):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax)
        ims.append(im)
        axes_line.append(ax)
        ax.set_xticks([])
        ax.set_yticks([])

        title = "GT" if j == 0 else method_names[j - 1]
        ax.set_title(title, fontsize=18)
        if j == 0:
            ax.set_ylabel("Vertical displacement", fontsize=18, rotation=90, labelpad=10)

    cbar = fig.colorbar(ims[-1], ax=axes_line, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.95)
    cbar.ax.tick_params(labelsize=10)

    data_list_strain = [results[implot]["deformation"]["gt"][0]] + [
        results[implot]["deformation"][m][0] for m in methods
    ]

    ims = []
    axes_line = []

    for j, data in enumerate(data_list_strain):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(data, cmap="inferno", vmin=0, vmax=vmaxstrain)
        ims.append(im)
        axes_line.append(ax)
        ax.set_xticks([])
        ax.set_yticks([])

        if j == 0:
            ax.set_ylabel("Deformation", fontsize=18, rotation=90, labelpad=10)

    cbar = fig.colorbar(ims[-1], ax=axes_line, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.95)
    cbar.ax.tick_params(labelsize=10)

    fields = [results[implot]["traction"]["gt"]] + [
        results[implot]["traction"][m] for m in methods
    ]

    quiv_line = None
    axes_line = []

    for j, field in enumerate(fields):
        ax = fig.add_subplot(gs[2, j])
        axes_line.append(ax)

        H, W = images[implot][0].shape[:2]
        y, x = np.mgrid[0:H:step, 0:W:step]

        u = field[0, ::step, ::step]
        v = field[1, ::step, ::step]
        norm = np.sqrt(u**2 + v**2)

        mask = (norm >= threshold_inf) & (norm <= threshold_sup)
        u = np.where(mask, u, 0)
        v = np.where(mask, v, 0)
        norm = np.sqrt(u**2 + v**2)

        ax.imshow(images[implot][0],
                  cmap='gray' if images[implot][0].ndim == 2 else None,
                  zorder=0)

        quiv_line = ax.quiver(
            x, y, v, u, norm,
            cmap='inferno', clim=(0, norm.max()),
            angles='xy', scale_units='xy', scale=scale, zorder=1
        )

        ax.axis('off')
        if j == 0:
            ax.text(-0.1, 0.5, "Traction force", fontsize=18, rotation=90,
                    va="center", ha="center", multialignment="center", transform=ax.transAxes)

    if quiv_line is not None:
        cbar = fig.colorbar(quiv_line, ax=axes_line, orientation="vertical",
                            fraction=0.03, pad=0.02, shrink=0.95)
        cbar.ax.tick_params(labelsize=10)

    if show: 
        plt.show()
        
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    

def save_scatter_comparison(dfs: pd.DataFrame, results_dir: Path):
    """
    Saves a comparison of the mean RMSE of the different optical flow methods present in the dfs dataframe 
    in estimating mechanical quanties. Displays the mean RMSE of the reconstructed displacement and strain fields
    and the RMSE of the reconstructed stress and boundary traction force fields

    Args:
        dfs (pandas.DataFrame): Dataframe containing the RMSE data
        results_dir (Path): Path to the folder where the graphs will be saved
    """
    desired_order = ["Proposed", "HS", "Farneback", "TV-L1", "ILK"]
    df_mean = pd.concat(dfs).groupby(level=0).mean().round(4)
    df_mean = df_mean.reindex([m for m in desired_order if m in df_mean.index])
    methods = df_mean.index.tolist()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    method_colors = {
        "Proposed": "blue",
        "HS": "orange",
        "Farneback": "green",
        "TV-L1": "purple",
        "ILK": "red",
    }

    method_markers = {
        "Proposed": "o",        
        "HS": "s", 
        "Farneback": "^", 
        "TV-L1": "^",
        "ILK": "D",
    }

    for method_key in methods:
        x = df_mean.loc[method_key, 'displacement']
        y = df_mean.loc[method_key, 'strain']
        axes[0].scatter(x, y, color=method_colors[method_key],marker=method_markers[method_key], s=90)
        if method_key=='Proposed':
            axes[0].text(
            x + 0.012, y - 0.0006,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        elif method_key=='HS': 
            axes[0].text(
            x - 0.006, y - 0.0007,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        elif method_key=='Farneback': 
            axes[0].text(
            x - 0.012, y - 0.0006,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        elif method_key=='ILK': 
            axes[0].text(
            x, y - 0.0025,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        else: 
            axes[0].text(
            x, y + 0.0005,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )

    axes[0].set_xlabel("RMSE Displacement")
    axes[0].set_ylabel("RMSE Strain")
    axes[0].grid(True)

    for method_key in methods:
        x = df_mean.loc[method_key, 'stress']
        y = df_mean.loc[method_key, 'traction force']
        axes[1].scatter(x, y, color=method_colors[method_key],marker=method_markers[method_key], s=90)
        if method_key=='Proposed':
            axes[1].text(
            x + 6, y - 0.15,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        elif method_key=='ILK': 
            axes[1].text(
            x - 3.5, y - 0.15,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )
        else: 
            axes[1].text(
            x, y + 0.2,
            method_key,
            fontsize=10,
            ha="center", va="bottom", fontweight="bold"
        )

    axes[1].set_xlabel("RMSE Stress")
    axes[1].set_ylabel("RMSE Traction force")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / 'plots' / 'scatter_rmse_mean', dpi=300)
    plt.close()


def save_table_rmse(rmse_table: pd.DataFrame, save_path: Path):
    """
    Saves the provided RMSE table in the form of a .png file

    Args:
        rmse_table (pd.DataFrame): RMSE table to save
        save_path (Path): Path to the folder where the tables will be saved
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Select only the columns you want to show
    subset = rmse_table[["displacement", "strain", "deformation", "stress", "traction force"]]

    # Then create the table
    table = ax.table(
        cellText=np.round(subset.values, 4),
        rowLabels=subset.index,
        colLabels=subset.columns,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    for key, cell in table.get_celld().items():
        cell.set_height(0.17)
        cell.set_width(0.2)
        if key[0] == 0:
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
        if key[1] == -1:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

def plot_reg(rmse_mean_reg: pd.DataFrame, factors_for_reg: List[float], results_dir:Path):
    """
    Saves a graph of the RMSE of the deformation and the traction for different
    optical-flow functions vs the factor used to scale the regularization parameters.

    Args:
        rmse_mean_reg (pd.DataFrame): RMSE table containing the relevant information
        factors_for_reg (List[float]): List of the factors scaling the regularization parameters
        results_dir (Path): Path to the folder where the tables will be saved
    """
    name_map = {
        "GT": "Ground Truth",
        "fista": "Proposed",
        "hs": "HS",
        "ilk": "ILK",
        "tv_l1": "TV-L1",
        "farneback": "Farneback",
    }

    methods = list(rmse_mean_reg['deformation'].keys())
    methods_plot = [(m, name_map.get(m, m)) for m in methods]

    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.flatten()
    titles = ['Deformation (regularization study)', 'Traction (regularization study)',
            'Deformation (noise study)', 'Traction (noise study)']
    keys_reg = ['deformation', 'traction']

    for i, (ax, key) in enumerate(zip(axes[:2], keys_reg)):
        for j, (method_key, method_label) in enumerate(methods_plot):
            ax.plot(
                factors_for_reg,
                rmse_mean_reg[key][method_key],
                linestyle=line_styles[j % len(line_styles)],
                marker=markers[j % len(markers)],
                linewidth=2.5,
                markersize=7,
                label=method_label
            )
        ax.set_xlabel('Factor for regularisation parameters', fontsize=25)
        ax.set_ylabel('', fontsize=15)
        ax.set_title(titles[i], fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=18)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='upper center', ncol=len(methods_plot))

    plt.savefig(results_dir / 'plots' / 'regularisation_study.png', dpi=300)
    plt.close()    
    
    
def plot_mean_error_noise(rmse_mean_noise: pd.DataFrame, stds: List[float], results_dir: Path):
    """
    Saves a graph of the RMSE of the deformation and the traction for different
    optical-flow functions vs the std of the noise applied to the original image.

    Args:
        rmse_mean_noise (pd.DataFrame): RMSE table containing the relevant information
        stds(List[float]): List of the stds of the gaussian noise applied to the image
        results_dir (Path): Path to the folder where the tables will be saved
    """
    name_map = {
    "GT": "Ground Truth",
    "fista": "Proposed",
    "hs": "HS",
    "ilk": "ILK",
    "tv_l1": "TV-L1",
    "farneback": "Farneback",
    }

    methods = list(rmse_mean_noise['deformation'].keys())

    methods_plot = [(m, name_map.get(m, m)) for m in methods]
    
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.flatten()
    titles = ['Deformation (regularization study)', 'Traction (regularization study)',
            'Deformation (noise study)', 'Traction (noise study)']
    keys_noise = ['deformation', 'traction']

    for i, (ax, key) in enumerate(zip(axes[:2], keys_noise)):
        for j, (method_key, method_label) in enumerate(methods_plot):
            ax.plot(
                stds,
                rmse_mean_noise[key][method_key],
                linestyle=line_styles[j % len(line_styles)],
                marker=markers[j % len(markers)],
                linewidth=2.5,
                markersize=7,
                label=method_label
            )
        ax.set_xlabel('Noise standard deviation', fontsize=25)
        ax.set_ylabel('', fontsize=12)
        ax.set_title(titles[i+2], fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=18)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='upper center', ncol=len(methods_plot))

    plt.savefig(results_dir / 'plots' / 'error_vs_noise.png', dpi=300)
    plt.close()
    

def plot_noise_reg(rmse_mean_reg: pd.DataFrame, factors_for_reg: List[float], rmse_mean_noise: pd.DataFrame, stds: List[float], results_dir: Path):
    """  
    Saves a graph of the RMSE of the deformation and the traction for different
    optical-flow functions vs the factor used to scale the regularization parameters, and 
    a graph of the RMSE of the deformation and the traction for different
    optical-flow functionsvs the std of the noise applied to the original image.

    Args:
        rmse_mean_reg (pd.DataFrame): RMSE table containing the relevant information for regularization study
        factors_for_reg (List[float]): List of the factors scaling the regularization parameters
        rmse_mean_noise (pd.DataFrame): RMSE table containing the relevant information for noise study
        stds(List[float]): List of the stds of the gaussian noise applied to the image
        results_dir (Path): Path to the folder where the tables will be savedSaves a graph of the RMSE of the deformation and the traction for different
            optical-flow functions vs the std of the noise applied to the original image.
    """
    name_map = {
        "GT": "Ground Truth",
        "fista": "Proposed",
        "hs": "HS",
        "ilk": "ILK",
        "tv_l1": "TV-L1",
        "farneback": "Farneback",
    }

    methods = list(rmse_mean_reg['deformation'].keys())
    methods_plot = [(m, name_map.get(m, m)) for m in methods]

    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    axes = axes.flatten()
    titles = ['Deformation (regularization study)', 'Traction (regularization study)',
            'Deformation (noise study)', 'Traction (noise study)']
    keys_reg = ['deformation', 'traction']
    keys_noise = ['deformation', 'traction']

    for i, (ax, key) in enumerate(zip(axes[:2], keys_reg)):
        for j, (method_key, method_label) in enumerate(methods_plot):
            ax.plot(
                factors_for_reg,
                rmse_mean_reg[key][method_key],
                linestyle=line_styles[j % len(line_styles)],
                marker=markers[j % len(markers)],
                linewidth=2.5,
                markersize=7,
                label=method_label
            )
        ax.set_xlabel('Factor for regularisation parameters', fontsize=25)
        ax.set_ylabel('', fontsize=15)
        ax.set_title(titles[i], fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=18)

    for i, (ax, key) in enumerate(zip(axes[2:], keys_noise)):
        for j, (method_key, method_label) in enumerate(methods_plot):
            ax.plot(
                stds,
                rmse_mean_noise[key][method_key],
                linestyle=line_styles[j % len(line_styles)],
                marker=markers[j % len(markers)],
                linewidth=2.5,
                markersize=7,
                label=method_label
            )
        ax.set_xlabel('Noise standard deviation', fontsize=25)
        ax.set_ylabel('', fontsize=12)
        ax.set_title(titles[i+2], fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=25, loc='upper center', ncol=len(methods_plot))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(results_dir / 'plots' / 'noise_regularisation_study.png', dpi=300)
    plt.close()
    
