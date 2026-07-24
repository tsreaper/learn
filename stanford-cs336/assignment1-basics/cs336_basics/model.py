from math import sqrt
import torch
from einops import einsum
from jaxtyping import Float


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Float[torch.Tensor, " d_out d_in"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if weights is None:
            self.weights = torch.nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
            std = sqrt(2 / (in_features + out_features))
            torch.nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)
        else:
            self.weights = torch.nn.Parameter(weights).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weights: Float[torch.Tensor, " vocab_size d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if weights is None:
            self.weights = torch.nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            )
            torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)
        else:
            self.weights = torch.nn.Parameter(weights).to(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        weights: Float[torch.Tensor, " d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        if weights is None:
            self.weights = torch.nn.Parameter(
                torch.ones(d_model, device=device, dtype=dtype)
            )
        else:
            self.weights = torch.nn.Parameter(weights).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = 1 / torch.sqrt(
            einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model + self.eps
        )
        result = einsum(x, rms, "... d_model, ... -> ... d_model")
        result = einsum(result, self.weights, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)


def silu(x: Float[torch.Tensor, " ..."]) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        w1_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
        w2_weight: Float[torch.Tensor, " d_model d_ff"] | None = None,
        w3_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = max(64, round(d_model * 8 / 3 / 64) * 64)

        self.w1 = Linear(d_model, d_ff, weights=w1_weight, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, weights=w2_weight, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, weights=w3_weight, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> torch.Tensor:
        return self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
