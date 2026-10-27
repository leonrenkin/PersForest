"""Installation hints for optional plotting dependencies."""

from importlib import import_module


def require_plotting() -> None:
    """Load the plotting dependencies or explain how to install the extra.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Import failures inside an
        installed dependency propagate with their original diagnostic.
    """
    try:
        import_module("matplotlib")
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise ImportError(
            'Plotting requires Matplotlib. '
            'Install it with `pip install "persforest[plot]"` '
            'or `pip install ".[plot]"` from the repository.'
        ) from error
