import warnings
from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from jax.tree import map as jt_map
from jaxtyping import Array
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config, DataSubset


class Batch(NamedTuple):
    dynamic: dict[str, Array]
    static: Array = None
    y: dict[str, Array] = {}

    def __getitem__(self, key):
        warnings.warn(
            f"dict-style access ('batch[\"{key}\"]') is deprecated. Use attribute access ('batch.{key}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)

    def to_jax(self):
        """Convert all numpy arrays to jax arrays."""
        jax_batch = jt_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, self)
        return jax_batch


def _get_basin_subbasin_dict(cfg: Config, subset: DataSubset):
    def read_file(fp) -> dict:
        df = pd.read_csv(fp, dtype=str)
        if "basin" not in df.columns or "subbasin" not in df.columns:
            raise ValueError(f"Subbasin file {fp} must contain 'basin' and 'subbasin' columns.")

        return df.groupby("basin")["subbasin"].apply(set).to_dict()

    match subset:
        case DataSubset.train:
            basin_subset_dict = read_file(cfg.train_basin_file)
        case DataSubset.test:
            basin_subset_dict = read_file(cfg.test_basin_file)
        case DataSubset.predict:
            train = read_file(cfg.train_basin_file)
            test = read_file(cfg.test_basin_file)
            basin_subset_dict = {
                b: train.get(b, set()) | test.get(b, set()) for b in train.keys() | test.keys()
            }
        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")

    return basin_subset_dict


def _get_time_slice(cfg: Config, subset: DataSubset):
    match subset:
        case DataSubset.train:
            start, end = tuple(cfg.train_date_range)
        case DataSubset.validate:
            start, end = tuple(cfg.validate_date_range)
        case DataSubset.test:
            start, end = tuple(cfg.test_date_range)
        case DataSubset.predict:
            ranges = [
                cfg.train_date_range,
                cfg.validate_date_range,
                cfg.test_date_range,
            ]
            start = min(r[0] for r in ranges)
            end = max(r[1] for r in ranges)

        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")
    return slice(start, end)


def get_train_ds_stats(cfg: Config, *, dynamic: bool = True, static: bool = True) -> dict:
    print("Calculating training statistics for encoding and normalization...")
    train_basin_dict = _get_basin_subbasin_dict(cfg, "train")
    train_basins = list(train_basin_dict.keys())
    train_subbasins = {x for s in train_basin_dict.values() for x in s}

    static_df = pd.read_parquet(cfg.attributes_file)
    subbasin_mask = static_df.index.get_level_values("subbasin").isin(train_subbasins)
    static_train_df = static_df[subbasin_mask]

    stats = {}
    if static:
        stats["s_encoding"] = _get_static_encoding(
            static_train_df, cfg.static_encoding.model_dump()
        )
        stats["s_scale"] = _calculate_scale_from_data(cfg, static_train_df, stats["s_encoding"])
    if dynamic:
        stats["d_encoding"] = _get_prescribed_encoding(cfg.dynamic_encoding.model_dump())
        stats["d_scale"] = _get_scale_from_precomp_stats(cfg, train_basins, stats["d_encoding"])

    return stats


def _get_prescribed_encoding(encoding):
    new_cols = {}
    # Handle categorical columns
    if encoding["categorical"]:
        for col, categories in encoding["categorical"].items():
            if not categories:
                raise ValueError(
                    f"Dynamic categorical variable '{col}' must either be excluded or have "
                    "its categories prescribed in the config."
                )
            new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    if encoding["bitmask"]:
        for col, bits in encoding["bitmask"].items():
            if not bits:
                raise ValueError(
                    f"Dynamic bitmask variable '{col}' must either be excluded or have "
                    "its bits prescribed in the config."
                )
            new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

    encoding["encoded_columns"] = new_cols
    return encoding


def _get_scale_from_precomp_stats(cfg: Config, basins: list[str], encoding: dict):
    """
    Calculates mean and std for a subset of basins by aggregating pre-computed stats.
    Uses pre-stored attributes in each basin's Zarr group.
    """
    aggregated_stats = defaultdict(
        lambda: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "log_sum": 0.0,
            "log_sum_sq": 0.0,
            "positive_count": 0,
        }
    )

    for basin_id in basins:
        try:
            basin_path = cfg.zarr_dir / basin_id
            z_group = zarr.open(basin_path, mode="r")
            basin_stats = z_group.attrs["normalization_stats"]

            for var, stats in basin_stats.items():
                agg = aggregated_stats[var]
                agg["count"] += stats["count"]
                agg["sum"] += stats["sum"]
                agg["sum_sq"] += stats["sum_sq"]
                agg["min"] = min(agg["min"], stats["min"])
                agg["max"] = max(agg["max"], stats["max"])
                agg["log_sum"] += stats["log_sum"]
                agg["log_sum_sq"] += stats["log_sum_sq"]
                agg["positive_count"] += stats["positive_count"]
        except Exception:
            raise KeyError(f"Failed to load normalization stats for {basin_path}")

    scale = {}
    all_vars = set(aggregated_stats.keys()).union(set(encoding["encoded_columns"]))

    for var in all_vars:
        scale[var] = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}

        if var in encoding["encoded_columns"]:
            scale[var]["encoded"] = True
            continue

        stats = aggregated_stats[var]

        if var in cfg.log_norm_cols:
            log_mean = stats["log_sum"] / stats["positive_count"]
            log_var = (stats["log_sum_sq"] / stats["positive_count"]) - log_mean**2
            log_std = np.sqrt(max(log_var, 1e-12))
            scale[var]["log_norm"] = True
            scale[var]["offset"] = log_mean
            scale[var]["scale"] = log_std
        elif var in cfg.no_norm_cols:
            scale[var]["offset"] = 0
            scale[var]["scale"] = 1
        elif var in cfg.range_norm_cols:
            min_val, max_val = stats["min"], stats["max"]
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = stats["sum"] / stats["count"]
            variance = (stats["sum_sq"] / stats["count"]) - (mean**2)
            std = np.sqrt(max(variance, 0))
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


def _get_static_encoding(df: pd.DataFrame, encoding: dict):
    new_cols = {}
    # Handle categorical columns
    for col, categories in encoding["categorical"].items():
        if not categories:
            categories = df[col].astype("category").unique()
            encoding["categorical"][col] = categories  # Add inferred categories to encoding
        new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    for col, bits in encoding["bitmask"].items():
        if not bits:
            bit_data = df[col].astype(int)
            max_val = bit_data.max()
            nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
            # Check which bits are active across the dataset
            bits = [n for n in range(nbits) if ((bit_data // 2**n) % 2).any()]
            encoding["bitmask"][col] = bits  # Add inferred bits to encoding
        new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

    encoding["encoded_columns"] = new_cols
    encoding["removed_columns"] = list(new_cols.keys())
    return encoding


def _calculate_scale_from_data(cfg: Config, df: pd.DataFrame, encoding: dict):
    """
    Calculate normalization directly from dataset values.
    """
    encoded_cols = set([c for cl in encoding["encoded_columns"] for c in cl])
    removed_cols = set(encoding["removed_columns"])
    all_cols = set(df.columns).union(encoded_cols) - removed_cols
    scale = {
        k: {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0} for k in all_cols
    }
    for var in all_cols:
        if var in encoded_cols:
            scale[var]["encoded"] = True
            continue
        if var in cfg.log_norm_cols:
            scale[var]["log_norm"] = True
            scale[var]["offset"] = np.nanmean(np.log1p(df[var]))
        elif var in cfg.range_norm_cols:
            min_val = float(df[var].min())
            max_val = float(df[var].max())
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = float(df[var].mean())
            std = float(df[var].std())
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


class MCFLIDataset(Dataset):
    def __init__(self, cfg: Config, subset: str | DataSubset, train_stats=None):
        self.PAD_SIZE = 150
        self.cfg = cfg

        if isinstance(subset, str):
            subset = DataSubset(subset)
        self.data_subset = subset

        self.basin_subset_dict = _get_basin_subbasin_dict(cfg, subset)
        self.basins = list(self.basin_subset_dict.keys())
        self.subbasins = {x for s in self.basin_subset_dict.values() for x in s}

        self.time_slice = _get_time_slice(cfg, subset)

        self._load_time_series_data()

    def _load_time_series_data(self):
        basin_list = []
        site_list = []
        x_list = []
        y_list = []
        hws_list = []
        mask_list = []

        features = self.cfg.features.dynamic["swot-river"]
        targets = self.cfg.features.target

        for b in tqdm(self.basins):
            ds = xr.open_zarr(self.cfg.zarr_dir / b)

            sites = list(self.basin_subset_dict[b])
            try:
                ds_subset = ds.sel(subbasin=sites, date=self.time_slice)
                x = ds_subset[features]
                y = ds_subset[targets]
            except KeyError:
                # Some basins do not have SWOT data (no observable rivers).
                # These shouldn't really be in our config'd sites but catch it here.
                continue

            valid_features = ~np.isnan(x[features[0]].values)
            valid_target = ~np.isnan(y.to_array().values[0, ...])
            valid_mask = valid_features & valid_target

            for s_idx in range(valid_mask.shape[1]):
                t_ids = np.where(valid_mask[:, s_idx])[0]
                if len(t_ids) == 0:
                    continue
                else:
                    subbasin_id = x.subbasin.values[s_idx]

                    x_arr = (
                        x.isel(date=t_ids, subbasin=s_idx).to_array().values.T
                    )  # [time, features]
                    y_arr = (
                        y.isel(date=t_ids, subbasin=s_idx).to_array().values.T
                    )  # [time, features]

                    pos_width = x_arr[:, features.index("width_river")] > 0
                    pos_slope = x_arr[:, features.index("slope_river")] > 0
                    valid_indices = np.where(pos_width & pos_slope)[0]

                    if len(valid_indices) < 5:
                        continue

                    # Sub-select only physically valid timesteps
                    x_arr = x_arr[valid_indices, :]
                    y_arr = y_arr[valid_indices, :]
                    n_samples, n_features = x_arr.shape

                    if n_samples > self.PAD_SIZE:
                        raise ValueError(
                            f"array padding size ({self.PAD_SIZE}) exceeded by {n_samples} samples."
                        )

                    # Padding logic
                    x_pad = np.zeros((self.PAD_SIZE, n_features), dtype=x_arr.dtype)
                    x_pad[:n_samples, :] = x_arr

                    mask = np.zeros((self.PAD_SIZE, n_features), dtype=bool)
                    mask[:n_samples, :] = 1

                    y_pad = np.zeros((self.PAD_SIZE, y_arr.shape[1]), dtype=y_arr.dtype)
                    y_pad[:n_samples, :] = y_arr

                    hws_cols = ["d_wse_river", "width_river", "slope_river"]
                    hws_ids = [features.index(c) for c in hws_cols]
                    hws_pad = x_pad[:, hws_ids]

                    basin_list.append(b)
                    site_list.append(subbasin_id)
                    x_list.append(x_pad)
                    y_list.append(y_pad)
                    hws_list.append(hws_pad)
                    mask_list.append(mask)

        self.sample_basins = basin_list
        self.sample_sites = site_list
        self.x = np.stack(x_list)
        self.y = np.stack(y_list)
        self.hws = np.stack(hws_list)
        self.mask = np.stack(mask_list)

    def __len__(self):
        return len(self.sample_basins)

    def __getitem__(self, idx: int):
        return self.sample_basins[idx]

    def denormalize(self, x: Array, name: str, smearing_cf: float = 1) -> Array:
        """
        Denormalizes a feature or target by its name.

        Args:
            x (Array): Normalized data.
            name (str): Name of the variable to denormalize.

        Returns:
            np.ndarray: Denormalized data.
        """
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            log_space = x * scale + offset
            return jnp.expm1(log_space) * smearing_cf

        else:
            return x * scale + offset

    def denormalize_std(self, mu_normalized: Array, sigma_normalized: Array, name: str) -> Array:
        """
        Denormalizes standard deviation. For log-normal, requires both mu and sigma.
        """
        offset = self.d_scale[name]["offset"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            # mu and sigma are in log-space (normalized)
            mu_log = mu_normalized + offset
            sigma_log = sigma_normalized

            # Log-normal variance in original space
            # https://en.wikipedia.org/wiki/Log-normal_distribution
            var_original = (jnp.exp(sigma_log**2) - 1) * jnp.exp(2 * mu_log + sigma_log**2)
            return jnp.sqrt(var_original)
        else:
            # Z-score and range-norm case
            scale = self.d_scale[name]["scale"]
            return sigma_normalized * scale

    def denormalize_targets(self, y_normalized: dict[str, Array]) -> dict[str, Array]:
        """
        Denormalizes the target variable(s).
        Returns:
            dict[str, Array]: Denormalized target data.
        """
        y_denorm = {}
        for t_name, t_norm in y_normalized.items():
            y_denorm[t_name] = self.denormalize(t_norm, name=t_name)

        return y_denorm

    def normalize(self, x: Array, name: str) -> Array:
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]
        encoded = self.d_scale[name]["encoded"]

        if encoded:
            raise ValueError(f"Cannot normalize an encoded column: {name}")

        if log_norm:
            # Resolve the module (np or jnp) from the input x
            xp = jnp if isinstance(x, jnp.ndarray) else np
            return (xp.log1p(x) - offset) / scale
        elif scale != 0:
            return (x - offset) / scale
        else:
            return x - offset
