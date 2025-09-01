"""MAP method adapter to the unified API."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from jax import Array

from pbnn.core.api import InferenceMethod, Posterior

from .optimizer import MAPConfig, fit_map


@dataclass
class MAP(InferenceMethod):
    """Method adapter so MAP conforms to the unified API."""

    logposterior_fn: Callable[[Dict[str, Any], Dict[str, Array]], Array]
    cfg: MAPConfig

    def fit(
        self,
        model,
        train_ds: Dict[str, Array],
        valid_ds: Optional[Dict[str, Array]] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> Posterior:
        """Fit MAP and return a Posterior object."""
        return fit_map(
            logposterior_fn=self.logposterior_fn,
            network=model,
            train_ds=train_ds,
            cfg=self.cfg,
        )
