from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import equinox as eqx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import logsumexp
from jax.random import PRNGKey, split
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from data import DataLoader, Batch, GraphBatch, TabularBatch
from models import BaseModel, ManningNet
from models.layers.heads import GMM


@eqx.filter_jit
def _model_map(model, batch: Batch, key: PRNGKeyArray) -> dict[str, Array | dict[str, Array]]:
    """Applies the model to a batch of data using jax.vmap."""
    y_pred = model(batch, key)
    return y_pred


def _direct_values(
    pred: Array, node_mask: Array, denorm_fn: Callable | None
) -> tuple[Array, Array]:
    # TODO Do we want to save anything more than the final prediction?
    if len(pred.shape) == 3:
        pred = pred[-1, ...]  # Grab the final prediction for now.
    pred = pred[..., 0][node_mask]  # Drop padded nodes
    pred = denorm_fn(pred) if denorm_fn else pred

    return np.asarray(pred)  # np to detach from jax


def _gmm_values(y_hat: dict[str, Array], node_mask: Array, scale: float, offset: float, log_norm:bool) -> Array:
    """
    Computes the unbiased arithmetic mean from the GMM components back to original units.

    y_hat: dict from GMM head.
    scale: dataset.d_scale['discharge']['scale']
    offset: dataset.d_scale['discharge']['offset']
    """
    mu_nat = y_hat["mu"][node_mask] * scale + offset
    sigma_nat = y_hat["sigma"][node_mask] * scale
    log_pi = y_hat["log_pi"][node_mask]

    if log_norm:
        # Component moments in log-space for Z = Y + 1 ~ LogNormal
        log_comp_E_Z = mu_nat + 0.5 * np.square(sigma_nat)
        log_comp_E_Z2 = 2.0 * mu_nat + 2.0 * np.square(sigma_nat)

        # Mixture moments in log-space
        log_E_Z = logsumexp(log_pi + log_comp_E_Z, axis=-1)
        log_E_Z2 = logsumexp(log_pi + log_comp_E_Z2, axis=-1)

        # Stable log-variance
        log_diff = np.minimum(2.0 * log_E_Z - log_E_Z2, -1e-16)
        log_var = log_E_Z2 + np.log(-np.expm1(log_diff))

        MAX_LOG_VAL = 700.0
        mixture_mean = np.exp(np.clip(log_E_Z, None, MAX_LOG_VAL)) - 1.0
        mixture_std = np.exp(np.clip(0.5 * log_var, None, MAX_LOG_VAL))

    else:
        # Each component is Gaussian in original space after unstandardizing.
        # Mixture mean: E[Y] = sum_k pi_k * mu_k
        pi = np.exp(log_pi)
        mixture_mean = np.sum(pi * mu_nat, axis=-1)

        # Mixture variance: Var[Y] = sum_k pi_k * (sigma_k^2 + mu_k^2) - mean^2
        # (law of total variance)
        E_Y2 = np.sum(pi * (np.square(sigma_nat) + np.square(mu_nat)), axis=-1)
        mixture_var = E_Y2 - np.square(mixture_mean)
        mixture_std = np.sqrt(np.maximum(mixture_var, 0.0))


    return mixture_mean, mixture_std


def model_iterate(
    model: BaseModel,
    dataloader: DataLoader,
    quiet: bool = False,
    denormalize: bool = True,
) -> Iterator[dict]:
    """Iterates through a dataloader and yields predictions from the model.

    Parameters
    ----------
    model: eqx.Module
        The model to be applied to the data.
    dataloader: Dataloader
        The dataloader providing batches of data.
    quiet: bool, optional
        If True, disables the progress bar. Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.

    Yields
    -------
    Iterator[dict]
        An iterator yielding dictionaries. Each dictionary contains the following keys:
            - 'basin': Basin IDs.
            - 'date': Dates.
            - 'y_pred': Model predictions.
            - 'y': Target values (if available).
            - 'dt': Dynamic time features (if available).
    """
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)
    key = PRNGKey(0)

    denorm_fns = {
        t: partial(dataloader.denormalize, name=t) if denormalize else None for t in model.target
    }

    for batch_meta, batch in tqdm(dataloader, disable=quiet):
        batch = batch.to_jax()
        key, pred_key = split(key)
        y_pred = _model_map(model, batch, pred_key)

        out_dict = {'metadata': batch_meta, 'batch': batch, 'y_pred': {}}

        if "vae_ood_score" in y_pred.keys():
            out_dict['ood_score'] = y_pred['vae_ood_score']

        for target_name in model.target:
            denorm_fn = denorm_fns[target_name]
            target_pred = y_pred[target_name]
            # Check the type of this target's head. Changes how we create and store predictions.
            if model.head is None:
                if isinstance(model, ManningNet):
                    summary = model.summarize_predictions(target_pred, batch.d_mask)
                    out_dict["y_pred"][target_name] = summary["Q_mean"]
                    out_dict["y_pred"][target_name + '_05'] = summary["Q_05"]
                    out_dict["y_pred"][target_name + '_95'] = summary["Q_95"]
                    out_dict["y_pred"][target_name + "_A0"] = summary["A0"]
                    out_dict["y_pred"][target_name + "_n"] = summary["n"]
                else:
                    raise NotImplementedError("model and head type combination not implemented for inference.")
            elif isinstance(model.head.get(target_name), GMM):
                scale = dataloader.dataset.d_scale[target_name]["scale"]
                offset = dataloader.dataset.d_scale[target_name]["offset"]
                log_norm = dataloader.dataset.d_scale[target_name]["log_norm"]
                t_exp, t_std = _gmm_values(target_pred, batch.node_mask, scale, offset, log_norm)
                out_dict["y_pred"][target_name] = t_exp
                out_dict["y_pred"][target_name + "_std"] = t_std
            else:
                t_exp = _direct_values(target_pred, batch.node_mask, denorm_fn)
                out_dict["y_pred"][target_name] = t_exp

            if batch.y is not None:
                if "y" not in out_dict.keys():
                    out_dict["y"] = {}

                # TODO Do we want to save anything more than the final prediction?
                if isinstance(batch, GraphBatch):
                    y = batch.y[target_name][-1, ..., 0][batch.node_mask]
                elif isinstance(batch, TabularBatch):
                    y = batch.y[target_name][batch.d_mask]

                y = denorm_fn(y) if denorm_fn else y
                out_dict["y"][target_name] = np.asarray(y)

        yield out_dict


def model_df_iterate(
    model: eqx.Module,
    dataloader: DataLoader,
    *,
    quiet: bool = False,
    denormalize: bool = True,
):
    # Iterate through batches from the dataloader
    for batch_results in model_iterate(model, dataloader, quiet=quiet, denormalize=denormalize):
        metadata = batch_results["metadata"]
        y_pred_dict = batch_results["y_pred"]
        y_obs_dict = batch_results.get("y", {})
        ood_score = batch_results.get("ood_score")

        if isinstance(model, ManningNet):
            mask = np.where(batch_results['batch'].d_mask.ravel())
            metadata = {
                k: v.ravel()[mask] for k, v in metadata.items()
            }

        # Construct MultiIndex from all metadata keys preserved in their original form
        # Ensures compatibility with both Graph and Tabular batches
        row_index = pd.MultiIndex.from_arrays(
            [np.asarray(v) for v in metadata.values()], 
            names=list(metadata.keys())
        )

        df_parts = {
            "pred": pd.DataFrame({k: v.ravel() for k, v in y_pred_dict.items()})
        }

        if ood_score is not None:
            df_parts["ood"] = pd.DataFrame({"score": ood_score.ravel()})

        if y_obs_dict:
            if isinstance(model, ManningNet):
                mask = batch_results['batch'].d_mask
                df_parts["obs"] = pd.DataFrame({k: model.stack_obs(v, mask) for k, v in y_obs_dict.items()})
            else:
                df_parts["obs"] = pd.DataFrame({k: v.ravel() for k, v in y_obs_dict.items()})

        df = pd.concat(df_parts, axis=1)
        df.index = row_index

        yield df


def predict(model: eqx.Module, dataloader, *, quiet: bool = False, denormalize: bool = True):
    """Generates predictions from a model and dataloader.

    The function iterates through the dataloader, applies the model to each batch, and
    collects the predictions, target values, and dynamic time features (if available)
    into lists. These lists are then concatenated into arrays and organized into a
    pandas DataFrame with a MultiIndex.

    Parameters
    ----------
    model: eqx.Module
        The model to use for generating predictions.
    dataloader: Dataloader
        The dataloader providing data for the model.
    quiet: bool, optional
        If True, suppresses the progress bar. Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the predictions, observations (if available), and dynamic
        time features (if available). The DataFrame has a MultiIndex with levels 'basin'
        and 'date'. The columns are also MultiIndexed with levels 'Type' and 'Feature'.
    """

    batch_dfs = []
    # Iterate through the dataset, make predictions and collect data in lists.
    for batch_df in model_df_iterate(model, dataloader, quiet, denormalize):
        batch_dfs.append(batch_df)

    df = pd.concat(batch_dfs)

    return df


def predict_to_parquet(
    model: eqx.Module,
    dataloader: DataLoader,
    output_path: Path | str,
    *,
    quiet: bool = False,
    denormalize: bool = True,
):
    """
    Generates predictions and streams them directly to a Parquet file.

    This function processes the dataset in batches, creates a pandas DataFrame for
    each batch, and appends it to a Parquet file on disk. This approach is
    memory-efficient as it avoids loading all predictions into memory at once.

    Parameters
    ----------
    model: eqx.Module
        The model to use for generating predictions.
    dataloader: DataLoader
        The dataloader providing data for the model.
    output_path: Union[str, Path]
        The path to the output Parquet file.
    quiet: bool, optional
        If True, suppresses the progress bar from the underlying iterator.
        Default is False.
    denormalize: bool, optional
        If True, denormalizes the predictions and target values. Default is True.
    """
    writer = None  # Initialize the Parquet writer

    try:
        # Iterate through batches from the dataloader
        for batch_df in model_df_iterate(model, dataloader, quiet=quiet, denormalize=denormalize):
            #  Clean NaNs (unobserved subbasins)
            df = batch_df.dropna(how="any")

            if df.empty:
                continue

            # Write the batch DataFrame to the Parquet file
            table = pa.Table.from_pandas(df)

            if writer is None:
                # For the first batch, create the writer with the table's schema
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

    except RuntimeError as e:
        if "DataLoader timed out" in str(e):
            print("predict_to_parquet ended with a timeout in dataloader. Continuing.")
            return  # Still runs the `finally` block
        else:
            raise e

    finally:
        # Ensure the writer is closed to finalize the file
        if writer:
            writer.close()
            if not quiet:
                print(f"Results successfully written to {output_path}")
