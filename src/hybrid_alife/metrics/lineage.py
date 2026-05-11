"""Lineage tracking and export.

Builds parent->child trees that survive across generations so we can report
the effective lineage count (Hill 1D) and export a tree to JSON / Newick for
downstream analysis. Designed to be cheap to update from the per-step
SimState.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np


@dataclass
class LineageTree:
    """Sparse parent map plus first/last-seen bookkeeping.

    We do not store full chronologies — just enough to reconstruct the
    ancestor chain of each surviving lineage. Set membership keeps this
    O(N_lineages_ever) memory.
    """

    parents: dict[int, int] = field(default_factory=dict)
    first_gen: dict[int, int] = field(default_factory=dict)
    last_gen: dict[int, int] = field(default_factory=dict)

    def observe(self, generation: int, lineage_ids: np.ndarray, parent_ids: np.ndarray) -> None:
        lid = np.asarray(lineage_ids).astype(np.int64)
        pid = np.asarray(parent_ids).astype(np.int64)
        for i in range(lid.size):
            li, pi = int(lid[i]), int(pid[i])
            if li in self.last_gen:
                self.last_gen[li] = generation
                continue
            self.parents[li] = pi
            self.first_gen[li] = generation
            self.last_gen[li] = generation

    def ancestry(self, lineage_id: int) -> list[int]:
        chain = [lineage_id]
        seen = {lineage_id}
        cur = lineage_id
        while cur in self.parents:
            p = self.parents[cur]
            if p == cur or p in seen:
                break
            chain.append(p)
            seen.add(p)
            cur = p
        return chain

    def depth(self, lineage_id: int) -> int:
        return len(self.ancestry(lineage_id)) - 1

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "parents": {str(k): v for k, v in self.parents.items()},
                    "first_gen": {str(k): v for k, v in self.first_gen.items()},
                    "last_gen": {str(k): v for k, v in self.last_gen.items()},
                }
            ),
            encoding="utf-8",
        )

    def summary(self, surviving: np.ndarray | None = None) -> dict[str, float]:
        if surviving is None:
            ids = list(self.last_gen.keys())
        else:
            ids = [int(x) for x in np.asarray(surviving)]
        depths = [self.depth(i) for i in ids] if ids else [0]
        return {
            "total_lineages_seen": float(len(self.first_gen)),
            "surviving_lineages": float(len(ids)),
            "mean_lineage_depth": float(np.mean(depths)),
            "max_lineage_depth": float(np.max(depths)),
        }
