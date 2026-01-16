"""import_utils.py
Small helpers to make local (same-folder) imports robust inside ComfyUI custom nodes.

This module is intentionally tiny and dependency-free.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def ensure_local_import(file_path: str, *, prepend: bool = True) -> str:
    """Ensure the directory containing *file_path* is present in sys.path.

    ComfyUI sometimes loads custom nodes in a way that does not guarantee the node
    folder is on sys.path. Several nodes in this repo import sibling modules
    (e.g. vision_llm_node.py). This helper makes that import reliable.

    Args:
        file_path: Usually pass __file__ from the caller module.
        prepend: If True, insert at sys.path[0]; else append.

    Returns:
        The normalized directory path that was inserted/ensured.
    """
    node_dir = os.path.dirname(os.path.abspath(file_path))
    # Normalize for stable comparisons across platforms
    node_dir = os.path.normpath(node_dir)

    if node_dir and node_dir not in sys.path:
        if prepend:
            sys.path.insert(0, node_dir)
        else:
            sys.path.append(node_dir)

    return node_dir
