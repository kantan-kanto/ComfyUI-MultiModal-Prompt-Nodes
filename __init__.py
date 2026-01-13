# ComfyUI-MultiModal-Prompt-Nodes
# Copyright (C) 2026 kantan-kanto (https://github.com/kantan-kanto)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

__version__ = "1.0.5"

from .qwen_nodes import NODE_CLASS_MAPPINGS as qNODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as qNODE_DISPLAY_NAME_MAPPINGS
from .wan_nodes import NODE_CLASS_MAPPINGS as wNODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as wNODE_DISPLAY_NAME_MAPPINGS
from .vision_llm_node import NODE_CLASS_MAPPINGS as vNODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as vNODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mappings in [qNODE_CLASS_MAPPINGS, wNODE_CLASS_MAPPINGS, vNODE_CLASS_MAPPINGS]:
    NODE_CLASS_MAPPINGS.update(mappings)

for mappings in [qNODE_DISPLAY_NAME_MAPPINGS, wNODE_DISPLAY_NAME_MAPPINGS, vNODE_DISPLAY_NAME_MAPPINGS]:
    NODE_DISPLAY_NAME_MAPPINGS.update(mappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

