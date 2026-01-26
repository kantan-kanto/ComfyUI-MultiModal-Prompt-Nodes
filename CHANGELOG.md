# Changelog

All notable changes to ComfyUI-MultiModal-Prompt-Nodes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Additional model support (Llama, Gemini, etc.)
- Workflow templates
- Performance optimization
- Extended documentation
