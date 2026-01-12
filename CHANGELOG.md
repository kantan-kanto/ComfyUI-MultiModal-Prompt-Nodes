# Changelog

All notable changes to ComfyUI-MultiModal-Prompt-Nodes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2026-01-12

### Author
- kantan-kanto (https://github.com/kantan-kanto) - Initial development and release

### Added
- Initial release of ComfyUI-MultiModal-Prompt-Nodes
- Vision LLM Node: Local GGUF vision language model support
  - Qwen2.5-VL and Qwen3-VL compatibility
  - Multi-image input (up to 3 images)
  - Multiple style presets and custom system prompts
- Qwen Image Edit Prompt Generator: Image editing prompt generation
  - Dynamic model selection (local GGUF + cloud API)
  - Manual mmproj selection for Qwen3-VL
  - Multi-image support via image2/image3 inputs
- Wan Video Prompt Generator: Video generation prompt optimization
  - Text-to-Video and Image-to-Video task support
  - Local Qwen3-VL integration
  - Wan2.2-specific prompt templates
- General Prompt Rewriter: Cloud API-based prompt enhancement
  - Multiple style presets (General, Flux, SDXL, MMAudio)
  - Aliyun Dashscope API integration

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
