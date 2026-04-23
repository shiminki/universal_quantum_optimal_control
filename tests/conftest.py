import sys
from pathlib import Path

# Make the package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

torch.manual_seed(0)
