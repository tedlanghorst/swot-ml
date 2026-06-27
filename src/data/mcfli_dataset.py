import xarray as xr
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from .cached_basingraphdataset import DynamicCacheManager, CachedBasinGraphDataset


class MCFLIDataset(CachedBasinGraphDataset):
    """
    Fully in-memory dataset for MCFLI parameter modeling.
    Caches dynamic, target, static features and metadata.
    """

    def __init__(
        self,
        cfg: Config,
        subset: str,
        hw_corr_thresh: float | None = None,
        median_w: bool = False,
        median_s: bool = False,
        overwrite: bool = False,
    ):
        cache_mgr = DynamicCacheManager(cfg)
        cache_dir = cache_mgr.create_cache(subset, overwrite)
        super().__init__(cfg, cache_dir, subset)

        if len(self.features.dynamic) > 1:
            raise ValueError("MCFLI dataset only accepts 1 dynamic feature source")
        self.source_name = list(cfg.features.dynamic.keys())[0]
        self.source_columns = self.features.dynamic[self.source_name]
        self.samples = []

        for basin, subbasins in tqdm(
            self.basin_subset_dict.items(), desc="Caching basins in memory"
        ):
            if not (self.cache_dir / basin).is_dir():
                print(f"{basin} not found in cache")
                continue

            with xr.open_zarr(self.cache_dir / basin, consolidated=True) as basin_ds:
                ds = basin_ds.sel(subbasin=basin_ds["subbasin"].isin(list(subbasins)))

                # Skip basins without required variables
                if not all(col in ds.data_vars for col in self.source_columns):
                    continue
                target_var = self.target[0]
                if target_var not in ds.data_vars:
                    continue

                ds = ds.load()

                x_dynamic = ds[self.source_columns]  # keep as DataArray for coords
                y = ds[self.target]

                valid_features = ~np.isnan(x_dynamic[self.source_columns[0]].values)
                valid_target = ~np.isnan(y.to_array().values[0, ...])
                valid_mask = valid_features & valid_target

                for s_idx in range(valid_mask.shape[1]):
                    t_ids = np.where(valid_mask[:, s_idx])[0]
                    if len(t_ids) == 0:
                        continue
                    else:
                        subbasin_id = x_dynamic.subbasin.values[s_idx]

                        x_s = (
                            self.x_s.sel(subbasin=subbasin_id)[["dis_m3_pyr"]]
                            .to_array()
                            .values.astype(np.float32)
                        )

                        # Extract raw (normalized) data for current reach
                        x_d_raw = x_dynamic.isel(date=t_ids, subbasin=s_idx).to_array().values.T
                        y_arr_raw = y.isel(date=t_ids, subbasin=s_idx).to_array().values.T

                        # Extract and denormalize width and slope for filtering
                        # Width is index 1, Slope is index 2 in hws_cols
                        h_denorm = self.denormalize(
                            x_dynamic["d_wse_river"].isel(date=t_ids, subbasin=s_idx).values,
                            "d_wse_river",
                        )
                        w_denorm = self.denormalize(
                            x_dynamic["width_river"].isel(date=t_ids, subbasin=s_idx).values,
                            "width_river",
                        )
                        s_denorm = self.denormalize(
                            x_dynamic["slope_river"].isel(date=t_ids, subbasin=s_idx).values,
                            "slope_river",
                        )

                        # Identify valid indices where denormalized width and slope > 0
                        physical_mask = (w_denorm > 0) & (s_denorm > 0)

                        if np.sum(physical_mask) < 10:
                            continue

                        h_mask = self.get_z_mask(h_denorm, 5)
                        w_mask = self.get_z_mask(w_denorm, 5)

                        valid_t_indices = np.where(physical_mask & h_mask & w_mask)[0]

                        if len(valid_t_indices) < 10:
                            continue

                        # Require h and w be positively correlated.
                        if hw_corr_thresh:
                            h_valid = h_denorm[valid_t_indices]
                            w_valid = w_denorm[valid_t_indices]
                            corr_matrix = np.corrcoef(h_valid, w_valid)
                            if np.isnan(corr_matrix[0, 1]) or corr_matrix[0, 1] <= hw_corr_thresh:
                                continue

                        # Sub-select only physically valid timesteps
                        x_d = x_d_raw[valid_t_indices]
                        y_arr = y_arr_raw[valid_t_indices]
                        n_samples, n_features = x_d.shape

                        if n_samples > self.cfg.pad_size:
                            raise ValueError(
                                f"self.cfg.pad_size ({self.cfg.pad_size}) exceeded by {n_samples} samples."
                            )

                        # Padding logic
                        x_pad = np.ones((self.cfg.pad_size, n_features), dtype=x_d.dtype)
                        x_pad[:n_samples, :] = x_d

                        mask = np.zeros((self.cfg.pad_size, n_features), dtype=np.int8)
                        mask[:n_samples, :] = 1

                        y_pad = np.ones((self.cfg.pad_size, y_arr.shape[1]), dtype=x_d.dtype)
                        y_pad[:n_samples, :] = y_arr

                        # Construct hws with denormalized values for valid indices
                        hws_cols = ["d_wse_river", "width_river", "slope_river"]
                        hws_arr = (
                            x_dynamic[hws_cols]
                            .isel(date=t_ids, subbasin=s_idx)
                            .to_array()
                            .values.T[valid_t_indices]
                        )

                        for i, col in enumerate(hws_cols):
                            hws_arr[:, i] = self.denormalize(hws_arr[:, i], col)

                        if median_w:
                            hws_arr[:, 1] = np.median(hws_arr[:, 1])

                        if median_s:
                            hws_arr[:, 2] = np.median(hws_arr[:, 2])

                        hws_pad = np.ones((self.cfg.pad_size, hws_arr.shape[1]), dtype=x_d.dtype)
                        hws_pad[:n_samples, :] = hws_arr

                        reach_sample = {
                            "basin": basin,
                            "subbasin": subbasin_id,
                            "x_d": x_pad,
                            "d_mask": mask,
                            "x_s": x_s,
                            "hws": hws_pad,
                            "y": y_pad,
                        }
                        self.samples.append(reach_sample)

    def get_z_mask(self, x, lim):
        log_x = np.log1p(x + np.abs(np.min(x)))
        mad = np.median(np.abs((log_x - np.median(log_x))))
        z = 0.6745 * (log_x - np.median(log_x)) / mad

        return z < lim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_flat_jax(samples):
    basin = [s["basin"] for s in samples]
    subbasin = [s["subbasin"] for s in samples]
    x_d = jnp.array([s["x_d"] for s in samples])
    d_mask = jnp.array([s["d_mask"] for s in samples])
    x_s = jnp.array([s["x_s"] for s in samples])
    hws = jnp.array([s["hws"] for s in samples])
    y = jnp.array([s["y"] for s in samples])
    return basin, subbasin, x_d, d_mask, x_s, hws, y


class MCFLIDataLoader(DataLoader):
    def __init__(self, cfg: Config, dataset: MCFLIDataset):
        super().__init__(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_flat_jax,
        )

    # Expose these dataset methods for convenience.
    def denormalize(self, x: Array, name: str):
        return self.dataset.denormalize(x, name)

    def denormalize_std(self, mu_norm: Array, sigma_norm: Array, name: str):
        return self.dataset.denormalize_std(mu_norm, sigma_norm, name)

    def denormalize_target(self, y_normalized: Array):
        return self.dataset.denormalize_target(y_normalized)
