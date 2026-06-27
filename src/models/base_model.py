import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from .layers import heads


class BaseModel(eqx.Module):
    head: dict[str, eqx.Module] | None
    target: list[str]

    def __init__(self, hidden_size: int, target: list | str, head: str, *, key: PRNGKeyArray):
        if not isinstance(target, (list, str)):
            raise ValueError(f"target must be a list or string but got: {type(target)}")
        if isinstance(target, str):
            target = [target]

        is_string = [isinstance(t, str) for t in target]
        if not all(is_string):
            raise ValueError(
                f"target must be a list of string(s) but got types: {[type(t) for t in target]}"
            )
        self.target = target

        match head.lower():
            case "linear":
                single_head = heads.Linear(hidden_size, key=key)
            case "mlp":
                single_head = heads.MLP(hidden_size, hidden_size * 2, key=key)
            case "gmm":
                single_head = heads.GMM(hidden_size, hidden_size * 2, 100, key=key)
            case "cmal":
                single_head = heads.CMAL(hidden_size, hidden_size * 2, 100, key=key)
            case "none":
                self.head = None
                return
            case _:
                raise NotImplementedError(f"{head} not implemented in BaseModel")

        # All heads are initialized as the same but will diverge during training.
        self.head = {t: single_head for t in target}

    def __call__(self, data, key: PRNGKeyArray | None) -> Array:
        raise NotImplementedError

    def add_assimimlator(self, source_name, num_source_features, source_network_sizes):
        raise NotImplementedError

    def finetune_update(self, **kwargs):
        raise NotImplementedError
