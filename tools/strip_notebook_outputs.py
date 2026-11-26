"""Git clean filter: commit notebooks without outputs or machine-specific metadata.

Reads a notebook on stdin and writes it to stdout with outputs, execution
counts, widget state and execution timings removed, and the kernel metadata
reduced to a generic Python 3 kernel. The working copy keeps its outputs; only
what Git stores (and what ``git diff`` compares) is cleaned. Uses only the
standard library.

Enable it once per clone, from the repository root::

    git config filter.strip-notebook-outputs.clean "python3 tools/strip_notebook_outputs.py"
    git add --renormalize '*.ipynb'

``.gitattributes`` assigns the filter to ``*.ipynb``. Clones without this
setting are unaffected. Input that is not a notebook is passed through
unchanged.
"""
import json
import sys

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
CELL_METADATA_DROPPED = ("execution", "ExecuteTime", "collapsed", "scrolled")


def strip(notebook: dict) -> dict:
    """Return ``notebook`` without outputs and machine-specific metadata."""
    metadata = notebook.setdefault("metadata", {})
    metadata.pop("widgets", None)
    metadata["kernelspec"] = dict(KERNELSPEC)
    metadata["language_info"] = {"name": "python"}
    for cell in notebook.get("cells", []):
        cell_metadata = cell.get("metadata", {})
        for key in CELL_METADATA_DROPPED:
            cell_metadata.pop(key, None)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


def main() -> None:
    """Filter stdin to stdout."""
    text = sys.stdin.buffer.read().decode("utf-8")
    try:
        notebook = json.loads(text)
        if not isinstance(notebook, dict) or "cells" not in notebook:
            raise ValueError("not a notebook")
    except ValueError:
        sys.stdout.buffer.write(text.encode("utf-8"))
        return
    # Jupyter's own layout: one-space indent, sorted keys, trailing newline.
    cleaned = json.dumps(strip(notebook), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(cleaned.encode("utf-8"))


if __name__ == "__main__":
    main()
