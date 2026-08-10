#!/usr/bin/env python3

# Required to run multiple processes alongside JAX
from multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError as e:
    if "context has already been set" in str(e):
        # Possibly set from environment variables.
        pass
    else:
        raise

import pickle
import sys
import traceback
from pathlib import Path

import pandas as pd
import typer

# import config, data, train, evaluate
from config import AssimConfig, Config, DataSubset
from data import get_dataloaders, kill_dataloaders
from train import Trainer


def train_from_config(cfg: Config, log_dir: Path | None = None):
    """Trains a model from a given configuration file.

    Parameters
    ----------
    cfg : Config
        The configuration object.
    trainer_kwargs : dict, optional
        Additional keyword arguments to pass to the Trainer constructor. This is useful
        to continue training from a save point or to set a custom logging directory.

    Returns
    -------
    cfg : Config
        The updated configuration object.
    trainer : Trainer
        The Trainer object after training. Contains the model and other useful attributes.
    dataset : CachedBasinGraphDataset
        The CachedBasinGraphDataset loaded for training.
    """

    subsets = [DataSubset.train]
    if cfg.validate_every_n_steps is not None:
        subsets.append(DataSubset.validate)

    dls = get_dataloaders(cfg, subsets)

    trainer = None
    if log_dir and log_dir.is_dir():
        trainer = Trainer.load_last_checkpoint(log_dir)
        # Could fail to load if nothing was saved.
        if trainer is not None:
            trainer.training_dl = dls["train"]
            trainer.validation_dl = dls.get("validate")  # default None

    # Make a new trainer if nothing was saved/loaded
    if trainer is None:
        trainer = Trainer(cfg, dls["train"], dls.get("validate"), log_dir=log_dir)

    trainer.start_training()
    kill_dataloaders(dls)

    return cfg, trainer


def train_from_config_ensemble(config_path: Path, ensemble_seed: int):
    """Trains a model for an ensemble by modifying the random seed.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    ensemble_seed : int
        Seed used to differentiate ensemble members. This seed is added to the random
        seed in the config that is used in model creation and dataloader shuffling.

    Returns
    -------
    cfg : dict
        The updated configuration dictionary.
    model : eqx.Module
        The trained model.
    log_dir : Path
        The directory where training logs and checkpoints were saved.
    dataset : CachedBasinGraphDataset
        The CachedBasinGraphDataset loaded for training.
    """
    cfg = Config.from_file(config_path)
    cfg.model_args.seed += ensemble_seed

    log_dir = config_path.parent / "base_models" / f"seed_{ensemble_seed:02d}"
    cfg, trainer, dataset = train_from_config(cfg, log_dir)

    return cfg, trainer.model, trainer.log_dir


def train_new_assimilator(run_or_state_dir: Path, assim_path: Path):
    # Get the pre-trained model and trainer
    trainer = Trainer.load_from_run_or_state(run_or_state_dir)

    # Update the config
    assim_cfg = AssimConfig.from_file(assim_path)
    trainer.cfg.add_assimilation(assim_cfg)
    cfg = trainer.cfg

    # Update the logging
    now_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    assim_log_dir = trainer.log_dir / f"{assim_path.stem}_{now_str}"
    trainer.new_logger(assim_log_dir)
    # Separates the copied training log from the new messages.
    new_feats = list(assim_cfg.new_features.keys())
    log_msg = f"{'=' * 10} Starting new assimilation: {new_feats} {'=' * 10}"
    trainer.logger.info("=" * len(log_msg))
    trainer.logger.info(log_msg)
    trainer.logger.info("=" * len(log_msg))

    # Create a new dataloader and update in trainer
    train_dl = get_dataloaders(cfg, [DataSubset.train])["train"]
    trainer.training_dl = train_dl

    # Update the model with the new assimilation modules
    new_model = trainer.model
    for new_group in assim_cfg.new_features.keys():
        n_group_features = len(train_dl.dataset.features.dynamic[new_group])
        group_network_sizes = assim_cfg.model_args.get(new_group, {})
        new_model.add_assimilator(new_group, n_group_features, group_network_sizes)
        trainer.cfg.model_args.sparse_sizes[new_group] = n_group_features
    trainer.replace_model(new_model)

    if assim_cfg.freeze_base:
        if getattr(new_model, "backbone_leaves", None):
            trainer.freeze_components(new_model.backbone_leaves)
        elif "get_backbone" in dir(new_model):
            trainer.freeze_components(new_model.get_backbone())
    else:
        print("Assim was config'd to freeze base model but base not defined.")

    trainer.start_training()

    return cfg, trainer.model, assim_log_dir


# def finetune(finetune_yml: Path):
#     """Fine-tunes a pre-trained model using a separate configuration file.
#
#     Parameters
#     ----------
#     finetune_yml : Path
#         Path to the YAML file containing fine-tuning parameters. This file should be in
#         the same directory as the original model run. It contains minimal parameters,
#         only those that are updated.
#
#     Returns
#     -------
#     cfg : dict
#         The updated configuration dictionary.
#     model : eqx.Module
#         The fine-tuned model.
#     log_dir : Path
#         The directory where training logs and checkpoints were saved.
#     dataset : CachedBasinGraphDataset
#         The CachedBasinGraphDataset loaded for training.
#     """
#     # Load the config and manipulate it a bit
#     run_dir = finetune_yml.parent
#     trainer = Trainer.load_last_checkpoint(run_dir)
#     cfg = trainer.cfg.copy()
#
#     # Read in the finetuning parameters
#     finetune = read_yml(finetune_yml)
#     cfg.num_steps = trainer.step + finetune.get('additional_steps', 0)
#     cfg.transition_begin = trainer.step if finetune.get('reset_lr') else 0
#     cfg.cfg_path = finetune_yml
#
#     # Insert these params directly.
#     cfg.update(finetune.get('config_update', {}))
#
#     trainer_kwargs = {'continue_from': run_dir}
#
#     model_finetune_kwargs = finetune.get('model_update', None)
#     cfg, trainer, dataset = train_from_config(cfg, trainer_kwargs,
#                                               model_finetune_kwargs)
#
#     return cfg, trainer.model, trainer.log_dir, dataset


def hyperparam_grid_search(config_path: Path, idx: int):
    """Performs a single hyperparameter grid search trial using k-fold
    cross-validation.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    idx : int
        Index of the current hyperparameter combination in the grid.

    Returns
    -------
    None. Results are saved to files within the specified directories.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """

    cfg = Config.from_file(config_path)
    cfg = cfg.update_from_grid(idx)

    k = 4
    for i in range(k):
        log_dir = config_path.parent / "trials" / f"index_{idx}" / f"fold_{i}"
        out_file = log_dir / "test_data.pkl"
        if out_file.is_file():
            print(f"Fold {i + 1} of {k} was already completed.")
            continue

        cfg.test_basin_file = f"metadata/site_lists/k_folds/test_{i}_{k}.txt"
        cfg.train_basin_file = f"metadata/site_lists/k_folds/train_{i}_{k}.txt"
        dls = get_dataloaders(cfg, [DataSubset.train])

        if log_dir.is_dir():
            trainer = Trainer.load_last_checkpoint(log_dir)
        else:
            trainer = Trainer(cfg, dls["train"], log_dir=log_dir)

        start_step = trainer.step
        while trainer.step < trainer.num_steps:
            start_step = trainer.step
            trainer.start_training()
            if trainer.step == start_step:
                break

        # Check if we actually did some training this run.
        if (start_step != trainer.step) or (not out_file.is_file()):
            eval_model(
                cfg,
                trainer.model,
                dls["train"].dataset,
                trainer.log_dir,
                run_predict=False,
                run_train=False,
                make_plots=False,
            )
        print(f"Fold {i + 1} of {k} done!")

        kill_dataloaders(dls)


# def hyperparam_smac_optimize(config_path: Path, n_workers: int, n_runs: int):
#     from train import update_smac_config, manual_smac_optimize
#     from smac.utils.configspace import get_config_hash

#     cfg = Config.from_file(config_path)

#     def target_fun(updates, seed):
#         local_cfg = Config.from_file(updates["cfg_path"])
#         local_cfg = update_smac_config(local_cfg, updates, seed)

#         trial_name = get_config_hash(updates)
#         path = Path(local_cfg.cfg_path)
#         log_dir = path.parent / "trials" / f"{path.stem}_{trial_name}_{seed}"

#         local_cfg, trainer, dataset = train_from_config(local_cfg, log_dir)

#         eval_model(
#             local_cfg,
#             trainer.model,
#             dataset,
#             trainer.log_dir,
#             run_test=True,
#             run_predict=False,
#             run_train=False,
#             make_plots=False,
#         )

#         # Weird to load instead of returning directly but this is a bandaid.
#         results_file = trainer.log_dir / "test_data.pkl"
#         with open(results_file, "rb") as f:
#             results, bulk_metrics, basin_metrics = pickle.load(f)

#         # return basin_metrics['flux']['RE'].median()
#         return basin_metrics[:]["RE"].median().mean()

#     manual_smac_optimize(cfg, n_workers, n_runs, target_fun)


# def calc_attributions(run_dir: Path):
#     from evaluate import save_all_intgrads, plot_average_attribution

#     trainer = Trainer.load_last_checkpoint(run_dir)
#     cfg = trainer.cfg

#     dataset = CachedBasinGraphDataset(cfg, DataSubset.test)
#     dataloader = CachedBasinGraphDataLoader(cfg, dataset)

#     save_dir = run_dir / "figures" / "attribution"
#     save_dir.mkdir(parents=True, exist_ok=True)

#     save_all_intgrads(cfg, trainer.model, dataloader, save_dir, m_steps=10)
#     plot_average_attribution(save_dir, dataset)


# def load_prediction_model(run_dir: Path, chunk_idx: int | None = None):
#     """Loads a pre-trained model and chunk of the dataset into memory.

#     Parameters
#     ----------
#     run_dir : Path
#         Path to the model training directory.
#     chunk_idx : int
#         Index of the basin chunking to select and predict on.

#     Returns
#     -------
#     cfg : dict
#         The model training configuration dictionary.
#     model : eqx.Module
#         The pre-trained model weights and structure.
#     predict_dataset : CachedBasinGraphDataset
#         A dataset containing a subset of the prediction domain.
#     eval_dir : Path
#         A directory for saving the results of this subset.
#     """
#     # cfg, model, _ = load_model(run_dir)
#     trainer = Trainer.load_last_checkpoint(run_dir)
#     cfg = trainer.cfg
#     cfg.shuffle = False  # No need to shuffle for inference

#     # Load the train dataset to get normalization stats.
#     # TODO could just have the testing dataset calculate the train stats directly
#     train_dataset = CachedBasinGraphDataset(cfg, DataSubset.train)
#     if chunk_idx:
#         cfg.test_basin_file = f"metadata/site_lists/predictions/chunk_{chunk_idx:02}.txt"

#     predict_dataset = CachedBasinGraphDataset(cfg, DataSubset.predict, train_ds=train_dataset)

#     eval_dir = run_dir / "inference"
#     eval_dir.mkdir(parents=True, exist_ok=True)

#     return cfg, trainer.model, predict_dataset, eval_dir


def make_all_plots(
    results: pd.DataFrame,
    bulk_metrics: dict,
    basin_metrics: pd.DataFrame,
    data_subset: DataSubset,
    log_dir: Path,
):
    """Generates and saves some accuracy plots for a given data subset.

    Parameters
    ----------
    cfg : Config
        The configuration object.
    results : pd.DataFrame
        The model predictions and observations.
    bulk_metrics : dict
        The bulk evaluation metrics.
    basin_metrics : pd.DataFrame
        DataFrame of basin-specific evaluation metrics.
    data_subset : str
        Name of the data subset (e.g., 'train', 'test').
    log_dir : Path
        The directory where logs and figures are saved.

    Returns
    -------
    None. Figures are saved to the specified directory.
    """
    from evaluate import basin_metric_histograms, mosaic_scatter

    fig_dir = log_dir / "figures" / data_subset.value
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = mosaic_scatter(results, bulk_metrics, str(log_dir))
    fig.savefig(fig_dir / "density_scatter.png", dpi=300)

    figs = basin_metric_histograms(basin_metrics)
    for target, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_metrics_hist.png", dpi=300)


def eval_model(
    cfg: Config,
    model,
    log_dir: Path,
    run_test: bool | str = True,
    run_predict: bool | str = True,
    run_train: bool | str = True,
    make_plots=True,
):
    """Evaluates a model on specified data subsets, calculates metrics, and generates plots.

    Parameters
    ----------
    cfg : Config
        The model training configuration object.
    model : eqx.Module
        The pre-trained model weights and structure.
    dataset : CachedBasinGraphDataset
        A dataset for estimating and evaluating.
    log_dir : Path
        The directory to save results in.
    run_test, run_predict, run_train : bool | str, optional
        Whether to run the respective phase. Defaults to True. If a string is provided, this will be used as the stem of the filepath for saving results.
    make_plots : bool, optional
        Whether to save generic accuracy plots. Defaults to True.

    Notes
    -----
    The `predict` subset does not generate plots, regardless of the value of `make_plots`. These would be identical to `test`, as 'predict' covers the same basins and time period as 'test' but without validation data.

    Saves
    -----
    Pickled results, bulk metrics, and basin metrics for each data subset. Plots in the `log_dir` directory if `make_plots` is `True`.
    """
    from evaluate import get_all_metrics, get_basin_metrics, predict_to_parquet

    results_dir = log_dir / "results"
    results_dir.mkdir(exist_ok=True)

    def eval_data_subset(data_subset: DataSubset, out_stem=None):
        data_subset_str = data_subset.value

        if isinstance(out_stem, bool):
            if out_stem:
                out_stem = data_subset_str
            else:
                return
        results_file = results_dir / f"{out_stem}_results.parquet"
        metrics_file = results_dir / f"{out_stem}_metrics.pkl"
        print(f"Evaluating {data_subset_str} subset and saving to: {results_file}")

        dls = get_dataloaders(cfg, [data_subset], {data_subset: False})
        predict_to_parquet(model, dls[data_subset], results_file, quiet=cfg.quiet)
        kill_dataloaders(dls)

        if data_subset == DataSubset.predict:
            return

        results = pd.read_parquet(results_file)
        bulk_metrics = get_all_metrics(results)
        basin_metrics = get_basin_metrics(results)
        with open(metrics_file, "wb") as f:
            pickle.dump((bulk_metrics, basin_metrics), f)

        if make_plots:
            make_all_plots(results, bulk_metrics, basin_metrics, data_subset, log_dir)

    eval_data_subset(DataSubset.test, run_test)
    # eval_data_subset(DataSubset.predict, run_predict)
    eval_data_subset(DataSubset.train, run_train)


# ┌────────────────────────────────┐ #
# │         Command Line           │ #
# └────────────────────────────────┘ #
app = typer.Typer(help="Train, test, or run predictions with the hydrological model.")


@app.command()
def train(config: Path):
    cfg = Config.from_file(config.resolve())
    cfg, trainer = train_from_config(cfg)
    eval_model(cfg, trainer.model, trainer.log_dir)


@app.command()
def train_ensemble(config_path: Path, ensemble_seed: int):
    config_path = config_path.resolve()
    cfg, model, eval_dir = train_from_config_ensemble(config_path, ensemble_seed)
    eval_model(cfg, model, eval_dir)


@app.command()
def continue_train(run_dir: Path):
    cfg = Config.from_file(run_dir / "config.json")
    cfg, trainer = train_from_config(cfg, run_dir)
    eval_model(cfg, trainer.model, trainer.log_dir)


# @app.command()
# def finetune(config: Path = typer.Option(..., exists=True, help="Path to fine-tuning config file.")):
#     cfg, model, eval_dir, dataset = finetune(config.resolve())
#     eval_model(cfg, model, dataset, eval_dir)


@app.command()
def add_assimilator(run_or_state_dir: Path, config: Path):
    cfg, model, eval_dir = train_new_assimilator(run_or_state_dir.resolve(), config.resolve())
    eval_model(cfg, model, eval_dir)


@app.command()
def grid_search(config_path: Path, grid_index: int):
    hyperparam_grid_search(config_path.resolve(), grid_index)


# @app.command()
# def smac_optimize(config_path: Path, smac_runs: int, smac_workers: int):
#     hyperparam_smac_optimize(config_path.resolve(), smac_workers, smac_runs)


@app.command()
def test(run_or_state_dir: Path):
    trainer = Trainer.load_from_run_or_state(run_or_state_dir)
    eval_model(trainer.cfg, trainer.model, trainer.log_dir)


# @app.command()
# def attribute(training_dir: Path):
#     calc_attributions(training_dir.resolve())


# @app.command()
# def predict(model_dir: Path, basin_chunk_index: int):
#     run_test = run_train = False
#     run_predict = f"chunk_{basin_chunk_index:02}"
#     cfg, model, dataset, eval_dir = load_prediction_model(model_dir.resolve(), basin_chunk_index)
#     eval_model(cfg, model, dataset, eval_dir, run_test, run_predict, run_train)


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    print("Done! Returning from python.")
    sys.exit(0)
