"""Smoke-level guard: the package imports and declares a version."""

import twl


def test_package_imports_and_has_version() -> None:
    assert isinstance(twl.__version__, str)
    assert twl.__version__
