# Changelog

All notable changes to ComfyUI-MultiModal-Prompt-Nodes will be documented in this file.


## Unreleased

- Added ComfyUI cancel support for local GGUF prompt generation
  - Watches ComfyUI interrupt requests during `llama-cpp-python` generation and calls `llm.abort()` when cancellation is requested
  - Propagates interrupted local Qwen, Wan, and Vision LLM runs as ComfyUI processing interrupts instead of wrapping them as regular runtime errors


## [1.0.12] - 2026-05-19

- Updated Qwen cloud API model selection
  - Prioritized Qwen3.6 API models, including `qwen3.6-plus` and `qwen3.6-flash`
  - Kept legacy Qwen API models selectable for existing workflow compatibility
  - Added UI deprecation notices that distinguish models announced offline since 2026-05-13 from models scheduled offline on 2026-07-13

- Improved API model handling
  - Normalized annotated UI model labels before sending requests to DashScope
  - Updated Qwen Image Edit API defaults to use `qwen3.6-plus`
  - Updated Wan I2V API validation to allow Qwen3.6 vision-capable models in addition to `qwen-vl-*` models

- Documentation
  - Documented the Qwen API model deprecation notice handling in README upgrade notes


## [1.0.11] - 2026-05-09

- Added support for Qwen3.6 local GGUF models
  - Added Qwen3.6 filename detection (`qwen36` / `qwen3.6`)
  - Routed Qwen3.6 through the existing `Qwen35ChatHandler` path
  - Updated Qwen3.5/3.6 mmproj requirement messages and logging
  - Set `image_min_tokens=1024` for Qwen3.5/3.6 chat handler initialization

- Improved local GGUF model discovery (`by @bongobongo2020`)
  - Added discovery of GGUF models in paths registered by ComfyUI through `extra_model_paths.yaml`
  - Uses `folder_paths.get_folder_paths("text_encoders")` and `folder_paths.get_folder_paths("llm")` when available
  - Preserves absolute paths for models stored outside the default ComfyUI `models` directory

- Improved mmproj auto-detection (`by @bongobongo2020`)
  - If no family-prefixed mmproj matches but the model directory contains exactly one `mmproj-*.gguf`, that file is used automatically
  - Ambiguous directories with multiple unmatched mmproj files still require manual selection

- Documentation
  - Updated README notes for extra model paths and mmproj fallback behavior


## [1.0.10] - 2026-04-02

- Added support for Qwen3.5 local GGUF models
  - Added Qwen3.5 model detection and proper handler selection (`Qwen35ChatHandler`)
  - Fixed incorrect fallback to `Qwen3VLChatHandler` for Qwen3.5 model names
  - Updated mmproj handling for Qwen3.5 (requirement checks and auto-detection flow)

- Improved post-run cleanup behavior for local model nodes
  - `VisionLLMNode`, `WanVideoPromptGenerator`, and `QwenImageEditPromptGenerator` now call `cleanup()` at the end of execution
  - Introduced `cleanup(finalize=False/True)` to separate regular unload from final teardown on process exit
  - Added safe manager re-initialization after cleanup for stable repeated runs


## [1.0.9] - 2026-03-15

- Expanded the search scope for local Qwen-family GGUF models

  - Added `models/text_encoders` and all subdirectories under both `models/LLM` and `models/text_encoders` to the search paths
  - Centralized model path resolution and mmproj resolution in `local_gguf_utils.py` to reduce duplicated logic

- Improved mmproj selection behavior
  - The UI now shows only mmproj files from the same directory as the selected GGUF model
  - When `mmproj = (Not required)` is selected, the node now explicitly switches to text-only mode to avoid unnecessary vision handler usage

- Strengthened the local prompt rewrite flow for Qwen and Wan
  - Added dedicated system prompts for `qwen_image`, `qwen_image_edit`, `wan_t2v`, and `wan_i2v`
  - Tightened prompt instructions to reduce verbose analysis-style responses and make it easier to return only the final prompt body
  - Added a second pass that preserves quoted text and normalizes the result to Simplified Chinese when Chinese output is requested but another language is returned

- Expanded Qwen Image Edit Prompt Generator
  - Made the `image` input optional so `Qwen-Image` can also be used for text-only prompt generation
  - Treated mmproj as not required when `Qwen-Image` is run locally without images
  - Increased local inference `max_tokens` and `n_ctx` to better support longer prompt generation
  
- Improved the robustness of Wan Video Prompt Generator
  - Added explicit validation errors when Image-to-Video is used without an input image

## [1.0.8] - 2026-02-09

- Fixed issue where `Qwen2.5-VL` were always loaded in text-only mode even when a valid mmproj file was specified.
  - Added vision chat handler support for `Qwen2.5-VL`
  - Enable vision mode automatically when supported model + mmproj are present

- Improved mmproj auto-detection logic
  - Auto-detect now selects mmproj files based on model family prefix (qwen2, qwen3) instead of arbitrary alphabetical fallback
  - Prevents incorrect mmproj selection when multiple mmproj files exist in the same directory

## [1.0.7] - 2026-01-26

### Fixed
- Fixed incorrect detection of Qwen3-VL when mmproj is set to (Not required).
  - Disabled automatic mmproj detection and prevented use of the VL handler in this case.
  - Updated GGUFModelManager.load_model and node-side mmproj interpretation to correctly respect (Not required).

## [1.0.6] - 2026-01-16

### Fixed
- Fixed an issue where incorrect **mmproj** could remain loaded when switching between Qwen3-VL GGUF models
  - Properly unload and reload GGUF models when model or mmproj changes
  - Prevent stale vision projectors from being reused across different Qwen3-VL models
- Improved **mmproj auto-detection** logic to avoid accidentally picking mmproj files from other models

### Changed
- Refined internal GGUF model lifecycle management for better stability when switching models (e.g. 8B ↔ 4B)
- Minor internal refactors to reduce state leakage in llama-cpp-python based vision models
- Improved README documentation for clarity and accuracy:
  - Clarified project scope as a **prompt generator for QwenImageEdit and Wan2.2**
  - Reorganized Credits and Dependencies to clearly separate derived works and external dependencies
  - Updated llama-cpp-python installation notes to reference the JamePeng fork documentation directly, avoiding incomplete or misleading installation instructions

### Added
- Added a `backends/` directory as a **structural placeholder**
  - This directory does not change behavior in v1.0.6
  - Reserved for future refactoring of Local GGUF and Cloud API backends without changing node interfaces

## [1.0.5] - 2026-01-13

### Removed
- General Prompt Rewriter (nodes.py) - Removed as it was unchanged from original ComfyUI-QwenPromptRewriter
  - Users should use the original ComfyUI-QwenPromptRewriter for this functionality

### Changed
- Updated documentation to reflect 3-node architecture
- Clarified project scope: focus on multimodal (vision + text) capabilities

## [1.0.4] - 2026-01-12

### Author
- kantan-kanto (https://github.com/kantan-kanto) - Initial development and release

### Added
- Initial release of ComfyUI-MultiModal-Prompt-Nodes
- Vision LLM Node: Local GGUF vision language model support
  - Qwen2.5-VL and Qwen3-VL compatibility
  - Multi-image input (batch support)
  - Multiple style presets (raw, default, detailed, concise, creative)
- Qwen Image Edit Prompt Generator: Image editing prompt generation
  - Dynamic model selection (local GGUF + cloud API)
  - Manual mmproj selection for Qwen3-VL
  - Multi-image support via image2/image3 inputs
- Wan Video Prompt Generator: Video generation prompt optimization
  - Text-to-Video and Image-to-Video task support
  - Local Qwen3-VL integration
  - Wan2.2-specific prompt templates

### Technical Details
- GPL-3.0 license (due to llama-cpp-python dependency)
- Unified category: `multimodal/prompt`
- Python 3.10+ support
- Comprehensive error handling and user feedback

## [Unreleased]

### Planned
- Workflow templates
- Extended documentation
