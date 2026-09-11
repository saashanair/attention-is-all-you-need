import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)  # if you run on GPU