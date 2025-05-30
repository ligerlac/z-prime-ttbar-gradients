"""Example script for fitting a model using Evermore and JAX,
where the model contains everemore parameters as well as simple ones."""

from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import evermore as evm
from functools import partial


def model(params: dict[str, evm.Parameter], hists: dict[str, Array]) -> dict[str, Array]:
    mu_modifier = params["mu"].scale()
    return mu_modifier(hists["signal"]) + hists["bkg"]


@jax.jit
def loss(params, hists, observation):
    hists["bkg"] = params["mu_bkg"] * hists["bkg"]
    expectations = model(params, hists)
    loss_val = (
        evm.pdf.Poisson(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    return -jnp.sum(loss_val)

hists = {
    "signal": jnp.array([3]),
    "bkg": jnp.array([10])
}

params = {
    "mu": evm.NormalParameter(value=1.0, lower=0.0, upper=10.0),  # type: ignore[arg-type]
    "mu_bkg": 0.
}

observation = jnp.array([37])
expectations = model(params, hists)

learning_rate = 0.001

grad_fn = jax.grad(loss, argnums=0)
for step in range(1000):
    if step % 100 == 0:
        loss_val = loss(params, hists, observation)
        print(f"{step=} - {loss_val=:.6f} - {params['mu'].value=}, {params['mu_bkg']=}")

    grads = grad_fn(params, hists, observation)

    for key, value in grads.items():
        if isinstance(value, evm.Parameter):
            params[key] = evm.Parameter(
                value=params[key].value - learning_rate * value.value,
                lower=params[key].lower,
                upper=params[key].upper
            )
        else:
            params[key] = params[key] - learning_rate * value
