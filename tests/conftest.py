from pathlib import Path

import pytest

from msgengine.catalogue import Catalogue

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "demo_catalogue.json"


@pytest.fixture(scope="session")
def cat() -> Catalogue:
    return Catalogue.load(CATALOGUE_PATH)
