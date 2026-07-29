"""The analysis path must not require the optional ``train`` extra.

`uv sync` installs no deep-learning stack, and `dvt-stats` must run in that
configuration.
These tests run in a subprocess so the check is not defeated by another test
having already imported torch.
"""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("torch", "torchvision", "timm", "pytorch_lightning", "cv2", "pytorch_grad_cam")

PROBE = """
import sys
import dvt_thrombus.cli
import dvt_thrombus.report
loaded = [name for name in {heavy!r} if name in sys.modules]
print(",".join(loaded))
"""


def test_analysis_entry_points_do_not_import_the_training_stack():
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(heavy=HEAVY)],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = [name for name in result.stdout.strip().split(",") if name]
    assert not loaded, (
        f"importing the analysis entry points pulled in {loaded}; dvt-stats would then "
        "require the optional 'train' extra"
    )
