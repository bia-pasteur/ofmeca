from mechanics.src.optical_flow.algorithms import fista_of, hs_of, farneback, tv_l1, ilk
from pathlib import Path
from types import SimpleNamespace
import yaml
import numpy as np 

PHYSICAL_LENGTH = 1.0
PIXEL_SIZE = 300 / (4 * PHYSICAL_LENGTH)

config_path = "/Users/josephinelahmani/Desktop/ofmeca/mechanics/configs/analysis.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
params_fista = SimpleNamespace(**cfg["optical_flow"]["fista"])
params_hs = SimpleNamespace(**cfg["optical_flow"]["hs"])
params_farneback = SimpleNamespace(**cfg["optical_flow"]["farneback"])
params_tvl1 = SimpleNamespace(**cfg["optical_flow"]["tvl1"])
params_ilk = SimpleNamespace(**cfg["optical_flow"]["ilk"])

base_folder = Path('/Users/josephinelahmani/Desktop/ofmeca/data/viscoelas')
output_root = Path(
    "/Users/josephinelahmani/Desktop/ofmeca/data/optical_flow_viscoelas"
)
output_root.mkdir(parents=True, exist_ok=True)


for ym in [1000.0, 1250.0, 1500.0, 2000.0]:
    for eta in [100.0, 200.0, 300.0]:
        for seed in [1, 2, 3, 4, 5]:
            
            config_name = f"T_100.0_E_{ym}_nu_0.3_eta_{eta}"
            config_out_dir = output_root / config_name
            config_out_dir.mkdir(parents=True, exist_ok=True)
        
            img_path = base_folder / f'T_100.0_E_{ym}_nu_0.3_eta_{eta}/{seed}_img.npy'
            img = np.load(img_path)

            h_fista = fista_of(img, params_fista, global_flow=True)
            h_hs = hs_of(img, params_hs, global_flow=True)
            h_farneback = farneback(img, params_farneback, global_flow=True)
            h_tvl1 = tv_l1(img, params_tvl1, global_flow=True)
            h_ilk = ilk(img, params_ilk, global_flow=True)

            mask = img[0] != 0

            h_fista_mask = h_fista * mask / PIXEL_SIZE
            h_hs_mask = h_hs * mask / PIXEL_SIZE
            h_farneback_mask = h_farneback * mask / PIXEL_SIZE
            h_tvl1_mask = h_tvl1 * mask / PIXEL_SIZE
            h_ilk_mask = h_ilk * mask / PIXEL_SIZE
            
            seed_out_dir = config_out_dir / f"seed_{seed}"
            seed_out_dir.mkdir(parents=True, exist_ok=True)

            np.save(seed_out_dir / "h_fista.npy", h_fista_mask)
            np.save(seed_out_dir / "h_hs.npy", h_hs_mask)
            np.save(seed_out_dir / "h_farneback.npy", h_farneback_mask)
            np.save(seed_out_dir / "h_tvl1.npy", h_tvl1_mask)
            np.save(seed_out_dir / "h_ilk.npy", h_ilk_mask)