import os
from typing import Iterator, List, Optional, Tuple

import folder_paths


LOCAL_LLM_SEARCH_DIRS: Tuple[str, ...] = ("LLM", "text_encoders")


def to_models_relative_path(path: str) -> str:
    return os.path.relpath(path, folder_paths.models_dir).replace("\\", "/")


def iter_local_gguf_files() -> Iterator[str]:
    for subdir in LOCAL_LLM_SEARCH_DIRS:
        root_dir = os.path.join(folder_paths.models_dir, subdir)
        if not os.path.isdir(root_dir):
            continue

        for current_dir, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith(".gguf"):
                    yield os.path.join(current_dir, file_name)


def discover_local_gguf_models(qwen_only: bool = False) -> List[str]:
    models = []

    for full_path in iter_local_gguf_files():
        file_name = os.path.basename(full_path)
        if file_name.startswith("mmproj"):
            continue
        if qwen_only and "qwen" not in file_name.lower():
            continue
        models.append(to_models_relative_path(full_path))

    return sorted(set(models), key=str.lower)


def discover_local_mmproj_files() -> List[str]:
    mmproj_files = []

    for full_path in iter_local_gguf_files():
        file_name = os.path.basename(full_path)
        if file_name.startswith("mmproj"):
            mmproj_files.append(to_models_relative_path(full_path))

    return sorted(set(mmproj_files), key=str.lower)


def resolve_local_gguf_path(relative_path: str) -> str:
    return os.path.normpath(os.path.join(folder_paths.models_dir, relative_path))


def resolve_mmproj_path_for_model(model_path: str, mmproj_selection: Optional[str]) -> Optional[str]:
    if mmproj_selection in (None, "(Auto-detect)"):
        return None
    if mmproj_selection == "(Not required)":
        return "(Not required)"

    requested_path = resolve_local_gguf_path(mmproj_selection)
    model_dir = os.path.normcase(os.path.dirname(model_path))

    if os.path.exists(requested_path):
        requested_dir = os.path.normcase(os.path.dirname(requested_path))
        if requested_dir == model_dir:
            return requested_path
        print(
            "[Vision LLM Node] Warning: Selected mmproj is not in the same directory as the model. "
            "Falling back to auto-detect."
        )
        return None

    same_dir_candidate = os.path.normpath(os.path.join(os.path.dirname(model_path), os.path.basename(mmproj_selection)))
    if os.path.exists(same_dir_candidate):
        return same_dir_candidate

    print(f"[Vision LLM Node] Warning: mmproj not found next to model: {mmproj_selection}")
    return None
