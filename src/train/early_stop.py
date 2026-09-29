import logging

import numpy as np

logger = logging.getLogger("training")


class EarlyStopper:
    """
    Implements standard early stopping based on validation loss.

    Stops training when the validation loss has not improved by at least a relative threshold for a given number of 'patience' epochs.

    Args:
        patience (int): How many epochs to wait after last time validation loss improved significantly. Default: 5.
        threshold (float): Minimum improvement required to consider the loss as having improved. Should be positive. e.g., 0.01 means a 1% decrease (rel) or a 0.01 decrease (abs). Default: 0.01.
        threshold_mode (str): "rel" measures improvement relative to the best loss, which is unstable when the loss is near zero or changes sign (e.g. NLL). "abs" measures it in the loss's own units. Default: "rel".
    """

    def __init__(self, *, patience: int = 5, threshold: float = 0.01, threshold_mode: str = "rel"):
        if patience < 0 or threshold < 0:
            raise ValueError("Patience and threshold must be non-negative.")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.loss_list = []

    def __call__(self, current_loss):
        """Update the loss list and determine if training should stop."""
        self.loss_list.append(current_loss)
        return self.check_stop()

    def check_stop(self):
        """Check whether we should stop training based on the loss list"""
        best_loss = np.inf
        stall_count = 0
        improvement = 0
        for loss in self.loss_list:
            if not np.isfinite(best_loss):
                best_loss = loss
                continue
            improvement = best_loss - loss
            if self.threshold_mode == "rel":
                improvement = improvement / abs(best_loss)
            if improvement > self.threshold:
                best_loss = loss
                stall_count = 0
            else:
                stall_count += 1
        if self.threshold_mode == "rel":
            improvement_str = f"{improvement * 100:0.02f}%"
        else:
            improvement_str = f"{improvement:0.04f}"
        if stall_count == 0:
            logger.info(
                f"EarlyStopper: Patience reset. New best loss {best_loss:0.04f}. "
                + f"Improvement was {improvement_str}."
            )
        else:
            logger.info(
                f"EarlyStopper: Patience {stall_count}/{self.patience}. "
                + f"Improvement was {improvement_str}."
            )
        return stall_count >= self.patience

    # --- Methods for Saving/Loading State ---

    def get_state(self):
        """Returns a dictionary containing the essential state."""
        return {
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "loss_list": self.loss_list,
        }

    @classmethod
    def from_state(cls, state):
        """Creates a new instance from a saved state dictionary."""
        try:
            # Create instance with saved params
            instance = cls(
                patience=state["patience"],
                threshold=state["threshold"],
                threshold_mode=state.get("threshold_mode", "rel"),
            )
            # Set the dynamic state
            instance.loss_list = state["loss_list"]
            return instance
        except KeyError as e:
            raise ValueError(f"Missing key in state dictionary for creating instance: {e}")
