import pytest

from poliseye.data import synthetic_presence


@pytest.fixture(scope="session")
def wide():
    return synthetic_presence(n_days=21, seed=1)
