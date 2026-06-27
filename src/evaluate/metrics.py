import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_percentage_error, r2_score

log_pad = 0.001


def get_basin_metrics(df: pd.DataFrame, disp: bool = False):
    """Calculates various metrics for each basin in a DataFrame.

    The function applies the :py:func:`get_all_metrics` function to each basin group in the DataFrame
    to calculate metrics such as R2, MAPE, nBias, etc. The results are organized into a new
    DataFrame with a MultiIndex for columns, where the first level represents the feature
    and the second level represents the metric.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame containing observations and predictions, with a MultiIndex including 'basin'.
    disp: bool, optional
        If True, prints a summary of the median metric values for each feature. Default is False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated metrics for each basin and feature.
        The columns are MultiIndexed with levels 'Feature' and 'Metric'.
    """
    if 'reach_id' in df.index.names:
        level_name = 'reach_id'
    elif 'subbasins' in df.index.names:
        level_name = 'subbasins'
    else:
        raise NotImplementedError("Cannot find a grouping index")

    per_basin_metrics = df.groupby(level=level_name).apply(get_all_metrics)

    # Initialize an empty DataFrame to store results with multi-level columns
    feature_names = df["obs"].columns
    metric_names = per_basin_metrics.iloc[0][feature_names[0]].keys()
    multi_index_columns = pd.MultiIndex.from_product(
        [feature_names, metric_names],
        names=["Feature", "Metric"],
    )
    results_df = pd.DataFrame(columns=multi_index_columns)

    # Populate the DataFrame with the metrics results
    for basin, basin_metrics in per_basin_metrics.items():
        for feature, metrics in basin_metrics.items():
            for metric_name, metric_value in metrics.items():
                results_df.loc[basin, (feature, metric_name)] = metric_value

    if disp:
        metrics_str = "Basin Metrics\n"
        for feature in feature_names:
            metrics_str += f"{feature.upper()}\n"
            for metric in metric_names:
                metrics_str += f"{metric}: {np.nanmedian(results_df[feature][metric]):0.4f}\n"
            metrics_str += "\n"
        print(metrics_str)

    return results_df


def get_all_metrics(df: pd.DataFrame, disp: bool = False):
    """Calculates various metrics for each feature in a DataFrame.

    The function calculates a range of metrics for each feature in the DataFrame,
    comparing observed values ('obs') with predicted values ('pred'). The metrics
    include R2, MAPE, nBias, etc.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame containing observations and predictions.
    disp: bool, optional
        If True, prints a summary of the calculated metrics for each feature.
        Default is False.

    Returns
    -------
    dict
        A dictionary where keys are feature names and values are dictionaries of metrics.
    """
    metric_fn_map = {
        "R2": r2_score,
        "spearman_r": calc_spearman_r,
        "pearson_r": calc_pearson_r,
        "NSE": calc_nse,
        "sigE": calc_sigE,
        "rRMSE": calc_rrmse,
        "MAPE": mean_absolute_percentage_error,
        "nBias": calc_nbias,
        "RE": calc_rel_err,
        "RB": calc_rel_bias,
        "Agreement": calc_agreement,
    }
    metrics = {}

    for feature in df["obs"].columns:
        y_raw = df[("obs", feature)].to_numpy()
        y_hat_raw = df[("pred", feature)].to_numpy()

        has_std = ("pred", f"{feature}_std") in df.columns
        if has_std:
            y_std_raw = df[("pred", f"{feature}_std")].to_numpy()
            mask = (y_raw > log_pad) & (y_hat_raw > log_pad) & ~np.isnan(y_std_raw)
            y_std = y_std_raw[mask]
        else:
            mask = (y_raw > log_pad) & (y_hat_raw > log_pad)

        y = y_raw[mask]
        y_hat = y_hat_raw[mask]
        num_obs = np.sum(mask)

        metrics[feature] = {"num_obs": num_obs}
        if num_obs > 8:
            metrics[feature].update({name: fn(y, y_hat) for name, fn in metric_fn_map.items()})
            kge, corr, alpha, beta = calc_kge_comp(y, y_hat)
            metrics[feature].update({"KGE": kge, "corr": corr, "alpha": alpha, "beta": beta})

            if has_std:
                metrics[feature].update({"nse_prob": calc_nse_prob(y, y_hat, y_std)})
                metrics[feature].update({"MSESS": calc_msess(y, y_hat, y_std)})
            else:
                metrics[feature].update({"nse_prob": np.nan, "MSESS": np.nan})

        else:
            metrics[feature].update({name: np.nan for name in metric_fn_map.keys()})
            metrics[feature].update(
                {"KGE": np.nan, "corr": np.nan, "alpha": np.nan, "beta": np.nan}
            )
            metrics[feature].update({"nse_prob": np.nan, "MSESS": np.nan})

    if disp:
        metrics_str = "Bulk Metrics\n"
        for feature, feature_metrics in metrics.items():
            metrics_str += f"{feature.upper()}\n"
            for metric, value in feature_metrics.items():
                metrics_str += f"{metric}: {value:0.4f}\n"
            metrics_str += "\n"
        print(metrics_str)

    return metrics


def calc_spearman_r(y, y_hat):
    return spearmanr(y, y_hat).statistic

def calc_pearson_r(y, y_hat):
    return pearsonr(y, y_hat).statistic

def calc_nse(y, y_hat):
    denominator = ((y - y.mean()) ** 2).sum()
    numerator = ((y - y_hat) ** 2).sum()
    if denominator == 0:
        return np.nan
    else:
        return 1 - (numerator / denominator)


def calc_nse_prob(y, y_hat, y_std):
    expected_sq_error = np.square(y - y_hat) + np.square(y_std)
    obs_variance = np.square(y - np.mean(y))
    sum_expected_sq_error = np.sum(expected_sq_error)
    sum_obs_variance = np.sum(obs_variance)
    if sum_obs_variance == 0:
        return np.nan
    return 1 - (sum_expected_sq_error / sum_obs_variance)


def calc_kge_comp(y, y_hat):
    corr = np.corrcoef(y, y_hat)[0, 1]
    mean_y = np.mean(y)
    mean_y_hat = np.mean(y_hat)
    std_y = np.std(y)
    std_y_hat = np.std(y_hat)
    if std_y == 0 or mean_y == 0:
        kge = corr = alpha = beta = np.nan
    else:
        alpha = std_y_hat / std_y
        beta = mean_y_hat / mean_y
        kge = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge, corr, alpha, beta


def calc_msess(y, y_hat, y_std):
    expected_sq_error = np.square(y - y_hat) + np.square(y_std)
    obs_variance = np.square(y - np.mean(y))

    sum_expected_sq_error = np.sum(expected_sq_error)
    baseline_expected_sq_error = 2 * np.sum(obs_variance)

    if baseline_expected_sq_error == 0:
        return np.nan

    return 1 - (sum_expected_sq_error / baseline_expected_sq_error)


def calc_sigE(y, y_hat):
    rel_err = (y_hat - y) / np.mean(y)
    rel_err = rel_err - np.mean(rel_err)  # debias
    return np.quantile(np.abs(rel_err), 0.67)


def calc_nbias(y, y_hat):
    nBias = np.median((y_hat - y) / y)
    return nBias


def calc_mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def calc_rel_err(y, y_hat):
    MdALQ = np.median(np.abs(np.log(y_hat / y)))
    return 100 * (np.exp(MdALQ) - 1)


def calc_rel_bias(y, y_hat):
    MdLQ = np.median(np.log(y_hat / y))
    sign = np.sign(MdLQ)
    mag = np.exp(np.abs(MdLQ)) - 1
    return 100 * sign * mag


def calc_rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


def calc_rrmse(y, y_hat):
    rmse = calc_rmse(y, y_hat)
    mean_y_hat = np.mean(y_hat)
    if mean_y_hat == 0:
        return np.nan
    else:
        return rmse / mean_y_hat * 100


def calc_lnse(y, y_hat):
    log_y = np.log(y)
    log_yhat = np.log(y_hat)
    return calc_nse(log_y, log_yhat)


def calc_agreement(y, y_hat):
    """https://www.nature.com/articles/srep19401"""
    corr = np.corrcoef(y, y_hat)[0, 1]
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * np.abs(np.mean((y - np.mean(y)) * (y_hat - np.mean(y_hat))))
    numerator = np.mean((y - y_hat) ** 2)
    denominator = np.var(y) + np.var(y_hat) + (np.mean(y) - np.mean(y_hat)) ** 2 + kappa
    if denominator == 0:
        return np.nan
    else:
        return 1 - (numerator / denominator)
