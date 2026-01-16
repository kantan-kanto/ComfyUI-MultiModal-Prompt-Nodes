# backends (placeholder)

This directory is intentionally added as a **structural placeholder**.

Future versions may move implementation details here, e.g.:

- `local_gguf.py`: Local GGUF backend (llama-cpp-python, mmproj handling, caching)
- `cloud_api.py`: Cloud/API backend (DashScope or other providers)
- `messages.py`: Shared message/image preprocessing utilities

**Important:** Node interfaces (ComfyUI INPUT/RETURN types) are intended to remain stable.
