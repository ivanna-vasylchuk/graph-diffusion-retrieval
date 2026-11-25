"""Перевірка, що пакет reldiff встановлено та імпортується."""

import reldiff


def test_import():
    """Пакет імпортується і має версію."""
    assert reldiff.__version__
