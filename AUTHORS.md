# Authors

## Primary Author

**kantan-kanto** ([@kantan-kanto](https://github.com/kantan-kanto)) - 2026

### Contributions
- Vision LLM Node (vision_llm_node.py)
  - Local GGUF vision language model integration
  - Qwen2.5-VL and Qwen3-VL support
  - Multi-image input functionality
  - Multiple style presets (default, detailed, concise, creative, raw)
  
- Qwen Image Edit Prompt Generator (qwen_nodes.py)
  - Dynamic model selection (local GGUF + cloud API)
  - Qwen3-VL manual mmproj selection
  - Multi-image editing support
  
- Wan Video Prompt Generator (wan_nodes.py)
  - Text-to-Video and Image-to-Video prompt optimization
  - Wan2.2-specific prompt templates
  - Local Qwen3-VL integration for video tasks

- Project Infrastructure
  - Repository structure and organization
  - GPL-3.0 licensing
  - Documentation (README, CHANGELOG, CONTRIBUTING)
  - Unified category system (`multimodal/prompt`)

## Based On / Derived From

This project builds upon and derives from the following GPL-3.0 licensed projects:

### ComfyUI-QwenPromptRewriter
- **Author**: lihaoyun6
- **Repository**: https://github.com/lihaoyun6/ComfyUI-QwenPromptRewriter
- **License**: GPL-3.0
- **Used in**: `qwen_nodes.py`, `wan_nodes.py`
- **Contributions**: API integration patterns, system prompt concepts, workflow design

### ComfyUI-QwenVL
- **Author**: 1038lab
- **Repository**: https://github.com/1038lab/ComfyUI-QwenVL
- **License**: GPL-3.0
- **Used in**: `vision_llm_node.py`
- **Contributions**: GGUF model loading concepts, vision LM integration patterns

## Acknowledgments

This project integrates with and depends on:

- **ComfyUI** - Custom node architecture and workflow system
- **llama-cpp-python** - GGUF model loading and inference (GPL-3.0, required dependency)
- **Aliyun Dashscope API** - Cloud-based LLM services for Qwen models
- **Qwen Models** - Vision language models from Alibaba Cloud

## License

This project is licensed under the GNU General Public License v3.0.

All derived works must also be distributed under GPL-3.0 or a compatible license.

See [LICENSE](LICENSE) file for full license text.
