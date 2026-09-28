from __future__ import annotations

import numpy as np

from scripts.analyze_poet_campaign import paired_comparison


def test_paired_comparison_reports_campaign_statistics() -> None:
    result = paired_comparison(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]))
    assert result["n_pairs"] == 2
    assert result["mean_difference"] == 1.5
    assert result["wins"] == 2
    assert result["ties"] == 0
    assert result["exact_paired_sign_permutation_p"] == 0.5
    assert result["cliffs_delta"] == 0.75
