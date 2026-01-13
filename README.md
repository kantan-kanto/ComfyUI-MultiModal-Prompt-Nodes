# ComfyUI-MultiModal-Prompt-Nodes

**Version:** 1.0.5  
**License:** GPL-3.0

Advanced multimodal prompt generation nodes for ComfyUI with local GGUF models (Qwen-VL) and cloud API support.

---

## Important Notes

### Language Recommendation for Optimal Results
Based on extensive testing, **Wan2.2** and **Qwen-Image-Edit** respond **significantly better to Chinese prompts than English prompts**. 

**Recommendation:** Set `target_language` to **"zh"** (Chinese) for best results with these models, even if your input is in English. The models will generate more coherent and instruction-following outputs.

### Vision Input Compatibility
Vision input support varies by model and llama-cpp-python version. See Installation section for detailed compatibility information. Results may vary based on your specific environment.

---

## Features

### 1. Vision LLM Node
- **Local GGUF support**: Run Qwen2.5-VL and Qwen3-VL models locally
- **Multi-image input**: Support batch image input via ComfyUI's batch nodes (e.g., Images Batch Multiple)
- **Flexible prompting styles**: 
  - `raw`: Direct LLM response without system prompt
  - `default`: Balanced prompt enhancement
  - `detailed`: Rich visual details (colors, textures, lighting, atmosphere)
  - `concise`: Minimal keywords, focused on core elements
  - `creative`: Artistic interpretation with unique perspectives
- **Device selection**: Simple CPU/GPU dropdown for hardware control
- **Auto-detect mmproj**: Automatic detection or manual selection for Qwen3-VL

### 2. Qwen Image Edit Prompt Generator
- **Dynamic model selection**: Auto-detect local GGUF models and cloud API models
- **Image editing prompts**: Specialized for Qwen-Image-Edit tasks
- **Manual mmproj selection**: Choose specific mmproj files or use auto-detect
- **Multi-image support**: Up to 3 images via optional inputs (image2/image3)
- **Unified interface**: Consistent parameter ordering and naming
- **API key management**: Centralized configuration via `api_key.txt`
- **Device control**: CPU/GPU selection for local models

### 3. Wan Video Prompt Generator
- **Video generation prompts**: Optimized for Wan2.2 text-to-video and image-to-video
- **Local Qwen3-VL integration**: Use local models for prompt enhancement
- **Task-specific optimization**: Separate prompts for T2V and I2V workflows
- **Extended token limit**: 2048 tokens to support longer Chinese prompts (600+ characters)
- **Device selection**: CPU/GPU dropdown for local model execution
- **Optimized for Chinese**: Better performance with Chinese language prompts

---

## Installation

### 1. Clone Repository

Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-MultiModal-Prompt-Nodes.git
```

### 2. Install Dependencies

```bash
cd ComfyUI-MultiModal-Prompt-Nodes
pip install -r requirements.txt
```

**Alternative manual installation:**
```bash
pip install dashscope pillow numpy
```

### 3. Install llama-cpp-python (REQUIRED for local models)

**Important:** Model compatibility varies by llama-cpp-python version. Based on my testing environment:

| Version | Qwen2.5-VL (Text) | Qwen2.5-VL (Vision) | Qwen3-VL | 
|---------|-------------------|---------------------|----------|
| 0.3.16 (official) | ✅ | ❌ | ❌ |
| 0.3.21+ (JamePeng fork) | ✅ | ❌* | ✅ |

***Note:** Vision input support may vary depending on your environment and configuration. In my setup, I have not been able to get vision input working with Qwen2.5-VL even with the JamePeng fork.

**Recommended Installation (JamePeng fork for Qwen3-VL support):**
```bash
pip install llama-cpp-python==0.3.21 --break-system-packages
```

**Source:** https://github.com/JamePeng/llama-cpp-python

**My Environment Results:**
- Official llama-cpp-python 0.3.16: Qwen2.5-VL text-only, no vision input, Qwen3-VL fails to load
- JamePeng fork 0.3.21+: Qwen3-VL works with vision input, Qwen2.5-VL text works but vision input still unavailable

⚠️ **Disclaimer:** Your results may differ depending on system configuration, GPU drivers, and other factors. If you encounter issues, please verify your environment setup and consider reporting compatibility details.

### 4. Place Models

Place your GGUF models in `ComfyUI/models/LLM/`:
```
ComfyUI/models/LLM/
├── Qwen3VL-4B-Q4_K_M.gguf
├── Qwen3VL-4B-Q8_0.gguf
├── mmproj-qwen3vl-4b-f16.gguf
└── ...
```

### 5. Configure API Key (Optional, for cloud models)

For cloud API usage, create `api_key.txt` in the node folder:
```
ComfyUI/custom_nodes/ComfyUI-MultiModal-Prompt-Nodes/api_key.txt
```

Add your Alibaba Cloud Dashscope API key to this file.

---

## Usage

### Vision LLM Node

**Inputs:**
- `prompt`: Text prompt to rewrite/enhance
- `style`: Prompt rewriting style
  - `raw`: Direct LLM response without system prompt (useful for custom prompting)
  - `default`: Balanced prompt enhancement
  - `detailed`: Rich visual details
  - `concise`: Minimal, focused keywords
  - `creative`: Artistic interpretation
- `target_language`: Output language (auto/en/zh)
- `model`: Select from auto-detected local GGUF models
- `mmproj`: mmproj file selection
  - `(Auto-detect)`: Automatically search for matching mmproj
  - `(Not required)`: For Qwen2.5-VL or text-only mode
  - Specific file: Manually select mmproj file
- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (0.0-2.0, default: 0.7)
- `device`: CPU or GPU execution
- `image` (optional): Input image for vision-language processing

**Example workflow:**
1. Load Vision LLM Node
2. Enter basic prompt: "a cat sitting on a windowsill"
3. Attach image via batch node (optional)
4. Select Qwen3-VL model
5. Choose `(Auto-detect)` for mmproj or select specific file
6. Select style: `default`
7. Set device: `CPU` or `GPU`
8. Run to get enhanced prompt

### Qwen Image Edit Prompt Generator

**Inputs:**
- `image`: Primary input image (required)
- `prompt`: Edit instruction or image description
- `prompt_style`: 
  - `Qwen-Image-Edit`: For image editing tasks
  - `Qwen-Image`: For general image understanding
- `target_language`: Output language (auto/zh/en)
- `llm_model`: Model selection
  - `Local: xxx`: Local GGUF models (auto-detected)
  - API models: qwen-vl-max, qwen-plus, etc.
- `mmproj`: mmproj file (required for local Qwen3-VL)
  - `(Auto-detect)`: Automatic detection
  - `(Not required)`: For API models or Qwen2.5-VL
  - Specific file: Manual selection
- `max_retries`: Retry attempts for API calls (default: 3)
- `device`: CPU/GPU selection for local models
- `save_tokens`: Compress images to save API tokens
- `image2/image3` (optional): Additional context images

**Use cases:**
- Image editing prompt generation
- Multi-image context prompts
- Style transfer descriptions
- Visual question answering

**Recommended settings:**
- For best results: Set `target_language` to `zh` (Chinese)
- Use local models for privacy, API models for quality
- Enable `save_tokens` when using API models

### Wan Video Prompt Generator

**Inputs:**
- `prompt`: Video scene description
- `task_type`: 
  - `Text-to-Video`: Generate video from text description
  - `Image-to-Video`: Generate video from image + text
- `target_language`: Output language (auto/zh/en)
- `llm_model`: Model selection
  - `Local: xxx`: Local GGUF models
  - API models: qwen-vl-max (for I2V), qwen-plus, etc.
- `mmproj`: mmproj selection (same as other nodes)
- `max_retries`: API retry attempts
- `device`: CPU/GPU for local models
- `save_tokens`: Image compression for API
- `image` (optional): Reference frame for I2V tasks

**Optimized for:**
- Wan2.2 video generation
- Temporal coherence descriptions
- Camera movement instructions
- Scene transitions

**Important notes:**
- **Use Chinese prompts** (`target_language: zh`) for best results
- Supports up to 600+ Chinese characters (2048 tokens)
- For I2V tasks, use `qwen-vl-*` models

**Example T2V workflow:**
1. Enter prompt: "一只猫在窗台上看风景" (A cat looking at scenery on a windowsill)
2. Set `task_type`: Text-to-Video
3. Set `target_language`: zh
4. Select model (local or API)
5. Run to get optimized video prompt

**Example I2V workflow:**
1. Attach input image
2. Enter motion description: "镜头慢慢推进" (Camera slowly zooms in)
3. Set `task_type`: Image-to-Video
4. Set `target_language`: zh
5. Ensure model supports vision (qwen-vl-*)
6. Run to get I2V prompt

---

## Model Compatibility

### Qwen2.5-VL (Integrated mmproj)
- ✅ Qwen2.5-VL-2B: Text-only in my environment
- ✅ Qwen2.5-VL-7B: Text-only in my environment
- ⚠️ mmproj integrated but vision input unavailable in my setup

### Qwen3-VL (Separate mmproj)
- ✅ Qwen3-VL-4B: Full vision support with JamePeng fork
- ✅ Qwen3-VL-7B: Full vision support with JamePeng fork
- ✅ Requires matching mmproj file

### Recommended Quantization
- **Q4_K_M**: Balanced quality/size (recommended for most users)
- **Q5_K_M**: Higher quality, larger size
- **Q8_0**: Maximum quality, largest size

### Model Sources
- Qwen models: https://huggingface.co/Qwen
- GGUF conversions: https://huggingface.co/models?search=qwen+gguf
- mmproj files: Usually bundled with GGUF conversions

---

## Configuration

### System Requirements
- **RAM**: 8GB+ (16GB recommended for 7B models)
- **Storage**: 3-8GB per model (depending on quantization)
- **GPU**: Optional (CPU execution supported)
  - NVIDIA GPU: CUDA support via llama-cpp-python
  - AMD GPU: ROCm support (requires specific build)
  - Intel Arc: Limited support, CPU recommended

### Performance Tips
1. **Use Q4_K_M quantization** for faster inference and lower memory usage
2. **Reduce max_tokens** if hitting memory limits
3. **Enable GPU** if you have compatible hardware (select `GPU` in device dropdown)
4. **Use CPU for stability** if encountering GPU issues
5. **Batch multiple requests** when possible for efficiency
6. **Close other applications** to free up RAM during inference

### Memory Usage Guide
| Model | Quantization | RAM Usage |
|-------|--------------|-----------|
| Qwen3-VL-4B | Q4_K_M | ~4-5GB |
| Qwen3-VL-4B | Q8_0 | ~7-8GB |
| Qwen3-VL-7B | Q4_K_M | ~6-7GB |
| Qwen3-VL-7B | Q8_0 | ~12-14GB |

---

## Troubleshooting

### Installation Issues

**Q: "No module named 'llama_cpp'" error**  
A: Install llama-cpp-python: `pip install llama-cpp-python==0.3.21 --break-system-packages`

**Q: pip install fails with "externally-managed-environment"**  
A: Use `--break-system-packages` flag or create a virtual environment

**Q: "Failed to load model" with Qwen3-VL**  
A: Ensure you're using llama-cpp-python 0.3.21+ (JamePeng fork). Version 0.3.16 doesn't support Qwen3-VL.

### Runtime Issues

**Q: "mmproj not specified" error**  
A: Select an mmproj file (or choose `(Auto-detect)`) in the mmproj dropdown for Qwen3-VL models

**Q: "No models found" in model dropdown**  
A: 
1. Place GGUF models in `ComfyUI/models/LLM/`
2. Restart ComfyUI
3. Verify file extensions are `.gguf`

**Q: Vision input not working with Qwen2.5-VL**  
A: This is a known issue in my environment. Qwen2.5-VL currently only supports text input. Use Qwen3-VL for vision capabilities.

**Q: Out of memory errors**  
A: 
1. Use smaller quantization (Q4_K_M instead of Q8_0)
2. Reduce `max_tokens` parameter
3. Close other applications
4. Use a smaller model (4B instead of 7B)

**Q: Slow inference on CPU**  
A: Normal for large models. Consider:
1. Q4_K_M quantization (faster than Q8_0)
2. Smaller models (4B faster than 7B)
3. GPU acceleration if available

**Q: "API_KEY is not set" error with local models**  
A: This error should only appear when using API models. If using local models (starting with "Local:"), this is a bug - please report it.

### Output Quality Issues

**Q: Wan2.2 output is incoherent or doesn't follow instructions**  
A: Set `target_language` to `zh` (Chinese). Wan2.2 performs **significantly better** with Chinese prompts, even if your input is in English.

**Q: Qwen-Image-Edit not understanding my edits**  
A: 
1. Use `target_language: zh` for better results
2. Be specific in edit instructions
3. Try using reference examples in your prompt

**Q: Output is cut off or incomplete**  
A: Increase `max_tokens` parameter (Vision LLM Node) or note that other nodes have fixed limits (512 for Qwen, 2048 for Wan)

### Device Selection Issues

**Q: How to choose between CPU and GPU?**  
A: 
- **GPU**: Faster inference, requires compatible hardware (NVIDIA with CUDA)
- **CPU**: Universal compatibility, slower but stable
- **Recommendation**: Start with CPU, switch to GPU if available and working

**Q: GPU selected but still using CPU**  
A: Your GPU may not be compatible with llama-cpp-python. Check:
1. NVIDIA GPU with CUDA support
2. llama-cpp-python built with CUDA support
3. Driver installation

---

## API Key Management

### For Cloud API Models

1. Create `api_key.txt` in the node directory:
```
ComfyUI/custom_nodes/ComfyUI-MultiModal-Prompt-Nodes/api_key.txt
```

2. Add your Alibaba Cloud Dashscope API key (single line, no quotes)

3. The key will be automatically loaded by Qwen and Wan nodes when using cloud API models

### Security Notes
- Never commit `api_key.txt` to version control
- The file is listed in `.gitignore` by default
- API keys are only loaded when using cloud API models
- Local models don't require API keys

---

## Examples

See the [examples/](examples/) directory for:
- Basic prompt enhancement workflows
- Multi-image vision processing
- Image editing prompt generation
- Video prompt generation (T2V and I2V)
- Style-specific optimizations

---

## License

This project is licensed under the **GNU General Public License v3.0**.

**Copyright (C) 2026 kantan-kanto**  
GitHub: https://github.com/kantan-kanto

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

**Note:** GPL-3.0 is required due to llama-cpp-python dependency.

For full details, see the [LICENSE](LICENSE) file and [AUTHORS.md](AUTHORS.md).

---

## Credits

### Original Authors
- **ComfyUI-QwenPromptRewriter**: [lihaoyun6](https://github.com/lihaoyun6/ComfyUI-QwenPromptRewriter) (GPL-3.0)
- **ComfyUI-QwenVL**: [1038lab](https://github.com/1038lab/ComfyUI-QwenVL) (GPL-3.0)

### Dependencies
- **llama-cpp-python**: [Andrei Betlen](https://github.com/abetlen/llama-cpp-python)
- **Qwen3-VL support**: [JamePeng's fork](https://github.com/JamePeng/llama-cpp-python)
- **Qwen models**: [Alibaba Cloud Qwen Team](https://github.com/QwenLM/Qwen)
- **Dashscope API**: Alibaba Cloud

For full attribution, see [AUTHORS.md](AUTHORS.md)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas needing help:
- Testing on different hardware configurations
- Documenting vision input compatibility across environments
- Additional workflow examples
- Performance optimizations

---

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: See [CHANGELOG.md](CHANGELOG.md) for version history
- **Examples**: Check [examples/](examples/) for workflow templates

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Current Version: 1.0.5
- Device selection: CPU/GPU dropdown
- Raw style for Vision LLM Node
- Unified interface across all nodes
- Extended token limit for Wan (2048)
- API key management via api_key.txt only
- mmproj auto-detect improvements
