"""Loss functions and factory.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def build_pn_loss() -> Callable:
    def pn_loss(logits: Tensor, labels: Tensor) -> Tensor:
        loss_fct = nn.Sigmoid()
        loss = loss_fct(-logits.view(-1) * labels.view(-1))
        return loss.mean()

    return pn_loss


def build_uu_loss(
    positive_prior: float = 0.5, theta_1: float = 0.65, theta_2: float = 0.35
) -> Callable:
    denominator = theta_1 - theta_2
    weight_positive = (
        theta_2 + positive_prior - 2 * theta_2 * positive_prior
    ) / denominator
    weight_negative = (
        theta_1 + positive_prior - 2 * theta_1 * positive_prior
    ) / denominator
    constant = (
        theta_2 * (1.0 - positive_prior) + (1.0 - theta_1) * positive_prior
    ) / denominator
    sigmoid_activation = nn.Sigmoid()

    def custom_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_mask = labels == 1
        negative_mask = labels == -1

        positive_risk = (
            sigmoid_activation(-logits[positive_mask])
            if positive_mask.any()
            else torch.tensor(0.0, device=logits.device)
        )
        negative_risk = (
            sigmoid_activation(logits[negative_mask])
            if negative_mask.any()
            else torch.tensor(0.0, device=logits.device)
        )

        loss = (
            weight_positive * positive_risk.mean()
            + weight_negative * negative_risk.mean()
            - constant
        )
        return loss

    return custom_loss


def generalized_leaky_relu(x: torch.Tensor, gamma: float = -0.01) -> torch.Tensor:
    return torch.where(x > 0, x, gamma * x)


def build_robust_uu_loss(
    positive_prior: float = 0.5,
    theta_1: float = 0.65,
    theta_2: float = 0.35,
    gamma: float = -0.01,
):
    denominator = theta_1 - theta_2

    a = ((1 - theta_2) * positive_prior) / denominator
    b = (theta_2 * (1 - positive_prior)) / denominator
    c = ((1 - theta_1) * positive_prior) / denominator
    d = (theta_1 * (1 - positive_prior)) / denominator

    sigmoid_activation = nn.Sigmoid()

    def custom_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        U_mask = labels == 1
        U_prime_mask = labels == -1

        if U_mask.any():
            logits_U = logits[U_mask]
            u_positive_risk = sigmoid_activation(-logits_U)
            u_negative_risk = sigmoid_activation(logits_U)
        else:
            u_positive_risk = torch.tensor(0.0, device=logits.device)
            u_negative_risk = torch.tensor(0.0, device=logits.device)

        if U_prime_mask.any():
            logits_U_prime = logits[U_prime_mask]
            u_prime_positive_risk = sigmoid_activation(-logits_U_prime)
            u_prime_negative_risk = sigmoid_activation(logits_U_prime)
        else:
            u_prime_positive_risk = torch.tensor(0.0, device=logits.device)
            u_prime_negative_risk = torch.tensor(0.0, device=logits.device)

        positive_risk = a * u_positive_risk.mean() - c * u_prime_positive_risk.mean()
        positive_risk = generalized_leaky_relu(positive_risk, gamma)

        negative_risk = d * u_prime_negative_risk.mean() - b * u_negative_risk.mean()
        negative_risk = generalized_leaky_relu(negative_risk, gamma)

        return positive_risk + negative_risk

    return custom_loss


def build_loss_fn(
    loss_type: str,
    positive_prior: float = 0.5,
    theta_1: float = 0.9,
    theta_2: float = 0.1,
    gamma: float = -0.01,
):
    if loss_type == "pn":
        return build_pn_loss()
    if loss_type == "uu":
        return build_uu_loss(positive_prior, theta_1, theta_2)
    if loss_type == "robust_uu":
        return build_robust_uu_loss(
            positive_prior=positive_prior,
            theta_1=theta_1,
            theta_2=theta_2,
            gamma=gamma,
        )
    raise ValueError(f"Invalid loss type: {loss_type}")
