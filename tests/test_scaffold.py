import sys

import ingen_pydev


def test_package_imports() -> None:
    assert ingen_pydev is not None


def test_python_version_is_311_or_newer() -> None:
    assert sys.version_info >= (3, 11)
