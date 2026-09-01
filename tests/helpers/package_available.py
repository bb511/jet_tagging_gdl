import platform
from importlib.metadata import PackageNotFoundError, version


def _package_available(package_name: str) -> bool:
    """Is *package_name* installed?

    Asked of the installed distributions rather than of `pkg_resources`, which recent setuptools no
    longer ships.
    """
    try:
        return version(package_name) is not None
    except PackageNotFoundError:
        return False


_IS_WINDOWS = platform.system() == "Windows"

