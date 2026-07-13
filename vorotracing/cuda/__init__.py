import torch
import torch.nn.functional as F

from ._backend import _C
from ._wrapper import (
    TraceVoroTracing,
    prefetch_octmap_adj,
    trace_octmap_infer,
    trace_octmap_infer_q8,
)

farthest_neighbor = _C.farthest_neighbor
NNBVH = _C.NNBVH
nearest_neighbor_bvh = _C.nearest_neighbor_bvh


def build_voronoi_adjacency(
    points: torch.Tensor,
    *,
    debug: bool = False,
    initial_guesses: torch.Tensor | None = None,
    initial_guesses_offsets: torch.Tensor | None = None,
    chunk_padding: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build standard Voronoi CSR adjacency with Paragram.

    VoroTracing's tracing kernels consume uint32 CSR tensors and assume a little
    tail padding on the flattened adjacency array. Paragram exposes the same CSR
    structure as int32 without padding, so this adapter keeps the model code
    independent from those representation details.
    """
    import paragram

    diagram = paragram.voronoi_diagram(
        points.detach(),
        debug=debug,
        initial_guesses=initial_guesses,
        initial_guesses_offsets=initial_guesses_offsets,
    )
    adjacency = F.pad(
        diagram.adjacency, (0, chunk_padding), mode="constant", value=0
    )
    return (
        adjacency.to(dtype=torch.uint32),
        diagram.offsets.to(dtype=torch.uint32),
        diagram.status,
    )


__all__ = [
    "TraceVoroTracing",
    "trace_octmap_infer",
    "trace_octmap_infer_q8",
    "prefetch_octmap_adj",
    "build_voronoi_adjacency",
    "farthest_neighbor",
    "NNBVH",
    "nearest_neighbor_bvh",
]
