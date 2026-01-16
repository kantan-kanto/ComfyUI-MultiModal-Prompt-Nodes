# ComfyUI-MultiModal-Prompt-Nodes
# Copyright (C) 2026 kantan-kanto (https://github.com/kantan-kanto)
# Based on ComfyUI-QwenVL by 1038lab
# Original: https://github.com/1038lab/ComfyUI-QwenVL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Vision LLM Node (vision_llm_node.py)
Local GGUF vision language model node for ComfyUI
Supports Qwen2.5-VL and Qwen3-VL with multi-image input
Part of ComfyUI-MultiModal-Prompt-Nodes
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any

import folder_paths

# llama-cpp-python imports
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    
    # Qwen2.5-VL and Qwen3-VL vision support via Qwen2VLChatHandler / Qwen25VLChatHandler
    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        QWEN3_AVAILABLE = True
    except ImportError:
        QWEN3_AVAILABLE = False
    
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    QWEN3_AVAILABLE = False
    print("[Vision LLM Node] Warning: llama-cpp-python not available")

# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPTS = {
    "default": """You are a professional prompt engineer. Enhance the user's input to create a more detailed and expressive prompt for image generation while maintaining the original intent.

Requirements:
1. Expand brief inputs with reasonable details
2. Enhance subject characteristics (appearance, expression, pose, etc.)
3. Add appropriate visual style and composition details
4. Keep the core meaning unchanged
5. Output should be clear and concise

Output the enhanced prompt directly without explanation.""",

    "detailed": """You are an expert prompt engineer specializing in detailed image descriptions.

Your task:
1. Analyze the user's input thoroughly
2. Add rich visual details (colors, textures, lighting, atmosphere)
3. Specify camera angles, composition, and framing
4. Include artistic style and mood
5. Maintain logical consistency and visual coherence

Create a comprehensive, vivid prompt that guides precise image generation.""",

    "concise": """You are a minimalist prompt engineer.

Your task:
1. Extract the core elements from user input
2. Use precise, impactful keywords
3. Remove redundancy while keeping essential details
4. Focus on main subject and key attributes
5. Output should be brief but complete

Create a clean, focused prompt.""",

    "creative": """You are a creative prompt engineer with artistic vision.

Your task:
1. Interpret user's intent creatively
2. Suggest interesting visual elements and compositions
3. Add artistic flair and unique perspectives
4. Enhance mood and atmosphere
5. Balance creativity with coherence

Transform the input into an inspiring, imaginative prompt.""",

    "raw": """You are a helpful assistant. Respond to the user's input directly and naturally."""
}

# ============================================================================
# Utility Functions
# ============================================================================

def encode_image_base64(pil_image: Image.Image, max_pixels: int = 262144) -> str:
    """
    Convert PIL image to base64 encoded string
    
    Args:
        pil_image: PIL Image object
        max_pixels: Maximum pixels for token saving
    
    Returns:
        Base64 encoded image string
    """
    # Resize image
    width, height = pil_image.size
    total_pixels = width * height
    
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # JPEG compress and base64 encode
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", optimize=True, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def tensor2pil(image_tensor) -> List[Image.Image]:
    """
    Convert ComfyUI tensor to PIL Image list
    
    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C)
    
    Returns:
        List of PIL Image objects
    """
    batch_count = image_tensor.size(0) if len(image_tensor.shape) > 3 else 1
    
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image_tensor[i]))
        return out
    
    # Convert to numpy array (0-255)
    np_image = np.clip(255.0 * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(np_image)]

def detect_language(text: str) -> str:
    """
    Detect language from text (simple)
    
    Args:
        text: Input text
    
    Returns:
        'zh' or 'en'
    """
    # Check for CJK character ranges
    cjk_ranges = [('\u4e00', '\u9fff')]
    
    for char in text:
        if any(start <= char <= end for start, end in cjk_ranges):
            return 'zh'
    
    return 'en'

# ============================================================================
# Model Manager
# ============================================================================

class GGUFModelManager:
    """GGUF model manager class

    Notes on robustness:
    - Qwen3-VL uses an mmproj (vision projector). In practice, llama-cpp / chat handlers
      may keep mmproj-related state alive longer than expected. To avoid "stale mmproj"
      issues when switching models (e.g., 8B -> 4B), we:
        * treat (model_path, mmproj_path, n_ctx, n_gpu_layers, vision_mode) as a signature
        * explicitly unload + gc before loading a different signature
        * make mmproj auto-detection choose the best match for the selected model
    """

    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler = None

        # Keep a full signature of what's currently loaded
        self._signature: Optional[tuple] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None

    def _normalize_path(self, p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return os.path.normpath(p)

    def _infer_is_qwen3(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        return "qwen3" in model_name_lower

    def _auto_detect_mmproj(self, model_path: str) -> Optional[str]:
        """Auto-detect an mmproj file that matches the selected model.

        Prefer mmproj names derived from the model base name; only fall back to other
        mmproj-*.gguf in the same directory if nothing matches.
        """
        model_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path)

        # Strip common suffixes
        if base_name.lower().endswith(".gguf"):
            base_name = base_name[:-5]

        # Candidates in preferred order (most specific first)
        preferred = [
            f"mmproj-{base_name}.gguf",
            f"mmproj-{base_name}-F16.gguf",
            f"mmproj-{base_name}-Q8_0.gguf",
        ]

        for fname in preferred:
            cand = os.path.join(model_dir, fname)
            if os.path.exists(cand):
                print(f"[GGUFModelManager] Auto-detected mmproj (preferred): {fname}")
                return self._normalize_path(cand)

        # Fallback: try any mmproj-*.gguf in the directory (stable sort for determinism)
        if os.path.exists(model_dir):
            mmprojs = sorted(
                [f for f in os.listdir(model_dir) if f.startswith("mmproj") and f.endswith(".gguf")]
            )
            if mmprojs:
                fname = mmprojs[0]
                cand = os.path.join(model_dir, fname)
                print(f"[GGUFModelManager] Auto-detected mmproj (fallback): {fname}")
                return self._normalize_path(cand)

        return None

    def _make_signature(
        self,
        model_path: str,
        mmproj_path: Optional[str],
        n_ctx: int,
        n_gpu_layers: int,
        use_vision: bool,
    ) -> tuple:
        return (
            self._normalize_path(model_path),
            self._normalize_path(mmproj_path),
            int(n_ctx),
            int(n_gpu_layers),
            bool(use_vision),
        )

    def load_model(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> Llama:
        """Load GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not available")

        model_path = self._normalize_path(model_path)

        # Infer Qwen version from model name
        is_qwen3 = self._infer_is_qwen3(model_path)

        # Qwen3-VL requires mmproj
        if is_qwen3:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "Qwen3-VL requires mmproj file!\n"
                        "Please download mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # Decide vision mode + initialize handler
        use_vision = False
        chat_handler = None

        if is_qwen3 and QWEN3_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Qwen3-VL with mmproj: {mmproj_path}")
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Qwen3-VL chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: Qwen3-VL requires an existing mmproj file")
                chat_handler = None
                use_vision = False
        elif "qwen2.5" in os.path.basename(model_path).lower() or "qwen2_5" in os.path.basename(model_path).lower():
            # Keep current behavior (text-only) to avoid changing features unexpectedly.
            print("[GGUFModelManager] Qwen2.5-VL: running in text-only mode (no chat handler)")
            chat_handler = None
            use_vision = False
        else:
            print("[GGUFModelManager] Using text-only mode")
            chat_handler = None
            use_vision = False

        new_sig = self._make_signature(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_vision=use_vision,
        )

        # If signature matches, reuse
        if self.model is not None and self._signature == new_sig:
            print(f"[GGUFModelManager] Using cached model: {model_path}")
            return self.model

        # Otherwise, explicitly unload to avoid stale mmproj/handler state
        if self.model is not None:
            print("[GGUFModelManager] Signature changed -> unloading previous model to avoid stale state")
            self.unload_model()

        print(f"[GGUFModelManager] Loading model: {model_path}")
        print(f"[GGUFModelManager] n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")

        # Store handler on manager (used later to decide if images are supported)
        self.chat_handler = chat_handler

        # Model loading
        if use_vision and self.chat_handler is not None:
            print("[GGUFModelManager] Loading with vision support")
            self.model = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                logits_all=True,  # required by some vision chat handlers
            )
        else:
            print("[GGUFModelManager] Loading in text-only mode")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

        self.current_model_path = model_path
        self.current_mmproj_path = self._normalize_path(mmproj_path)
        self._signature = new_sig

        print("[GGUFModelManager] Model loaded successfully")
        return self.model

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            print(f"[GGUFModelManager] Unloading model: {self.current_model_path}")
        try:
            if self.model is not None:
                del self.model
        finally:
            self.model = None

        try:
            if self.chat_handler is not None:
                del self.chat_handler
        finally:
            self.chat_handler = None

        self.current_model_path = None
        self.current_mmproj_path = None
        self._signature = None

        # Encourage timely cleanup (important for llama-cpp backends and mmproj state)
        import gc as _gc
        _gc.collect()

# Global model manager
_model_manager = GGUFModelManager()

# ============================================================================
# Prompt Rewriter Function
# ============================================================================

def rewrite_prompt_with_gguf(
    prompt: str,
    model_path: str,
    mmproj_path: Optional[str] = None,
    style: str = "default",
    target_language: str = "auto",
    images: Optional[List[Image.Image]] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
) -> str:
    """
    Rewrite prompt using GGUF model
    
    Args:
        prompt: Original prompt
        model_path: Path to GGUF model
        mmproj_path: Path to mmproj file (required for Qwen3-VL)
        style: Prompt style
        target_language: Target language
        images: Input image list (optional)
        max_tokens: Maximum tokens to generate
        temperature: Temperature parameter
        n_ctx: Context size
        n_gpu_layers: Number of GPU layers
    
    Returns:
        Rewritten prompt
    """
    # Load model
    model = _model_manager.load_model(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
    
    # detection
    detected_lang = detect_language(prompt)
    lang = target_language if target_language != "auto" else detected_lang
    
    # selection
    system_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["default"])
    

    if lang == "zh":
        prompt = f"[请用中文输出] {prompt}"
    elif lang == "en":
        prompt = f"[Please output in English] {prompt}"
    
    # construction
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    

    if images is not None and len(images) > 0:
        # Vision for
        if _model_manager.chat_handler is None:
            print("[Vision LLM Node] Warning: Images provided but vision is not available")
            print("[Vision LLM Node] Running in text-only mode, images will be ignored")
            print("[Vision LLM Node] To use vision features, please:")
            print("[Vision LLM Node]   1. Upgrade llama-cpp-python to latest version, OR")
            print("[Vision LLM Node]   2. Use Ollama instead")
        else:
            # Vision for
            user_content = []
            
            for img in images:
                img_b64 = encode_image_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            user_content.append({"type": "text", "text": prompt})
            messages[1]["content"] = user_content
    

    print(f"[Vision LLM Node] Generating with style='{style}', lang='{lang}'...")
    
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    

    result = response['choices'][0]['message']['content']
    result = result.strip()
    
    print(f"[Vision LLM Node] Original: {prompt[:100]}...")
    print(f"[Vision LLM Node] Enhanced: {result[:100]}...")
    
    return result

# ============================================================================
# ComfyUI Node
# ============================================================================

class VisionLLMNode:
    """
    Vision LLM Node - Local GGUF vision language models
    """
    
    @classmethod
    def INPUT_TYPES(cls):

        models_dir = os.path.join(folder_paths.models_dir, "LLM")
        
        available_models = []
        available_mmprojs = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".gguf"):
                    if file.startswith("mmproj"):
                        available_mmprojs.append(file)
                    else:
                        available_models.append(file)
        
        # mmproj selection
        mmproj_options = available_mmprojs + ["(Auto-detect)", "(Not required)"]
        if not available_mmprojs:
            mmproj_options = ["(Auto-detect)", "(Not required)"]
        

        if not available_models:
            available_models = ["(No GGUF models found in models/LLM/)"]
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Original prompt to be enhanced"
                }),
                "style": (["raw", "default", "detailed", "concise", "creative"], {
                    "default": "raw",
                    "tooltip": "Prompt rewriting style (raw: no system prompt, direct LLM response)"
                }),
                "target_language": (["auto", "en", "zh"], {
                    "default": "auto",
                    "tooltip": "Target language for output prompt"
                }),
                "model": (available_models, {
                    "default": available_models[0],
                    "tooltip": "GGUF model to use for rewriting"
                }),
                "mmproj": (mmproj_options, {
                    "default": mmproj_options[0],
                    "tooltip": "mmproj file (required for Qwen3-VL, select manually or use auto-detect)"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Maximum tokens to generate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature"
                }),
                "device": (["CPU", "GPU"], {
                    "default": "CPU",
                    "tooltip": "Device to run model on (GPU requires compatible hardware)"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision-based enhancement"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "rewrite"
    CATEGORY = "multimodal/prompt"
    DESCRIPTION = "Local GGUF vision-language models (Qwen3-VL, Qwen2.5-VL) for prompt generation"
    
    def rewrite(self, prompt: str, model: str, mmproj: str, style: str, target_language: str,
                max_tokens: int, temperature: float, device: str, image=None) -> tuple:
        """
        Rewrite prompt using GGUF model
        
        Returns:
            (enhanced_prompt,)
        """


        # llama-cpp-python check
        if not LLAMA_CPP_AVAILABLE:
            print("[Vision LLM Node] Error: llama-cpp-python not available")
            return (prompt,)
        

        if "(No GGUF models found" in model:
            print("[Vision LLM Node] Error: No GGUF models found in models/LLM/")
            return (prompt,)
        
        try:
            # construction
            models_dir = os.path.join(folder_paths.models_dir, "LLM")
            model_path = os.path.join(models_dir, model)
            
            if not os.path.exists(model_path):
                print(f"[Vision LLM Node] Error: Model not found: {model_path}")
                return (prompt,)
            
            # mmproj processing
            mmproj_path = None
            if mmproj not in ["(Auto-detect)", "(Not required)"]:
                mmproj_path = os.path.normpath(os.path.join(models_dir, mmproj))
                if not os.path.exists(mmproj_path):
                    print(f"[Vision LLM Node] Warning: mmproj not found: {mmproj_path}")
                    mmproj_path = None  # Auto-detect
            
            # PIL
            pil_images = None
            if image is not None:
                pil_images = tensor2pil(image)
                print(f"[Vision LLM Node] Using {len(pil_images)} image(s)")
            
            # Convert device selection to n_gpu_layers
            # GPU: -1 (all layers), CPU: 0 (no GPU)
            n_gpu_layers = -1 if device == "GPU" else 0
            
            # Load model
            print(f"[Vision LLM Node] Loading model: {model}")
            try:
                enhanced_prompt = rewrite_prompt_with_gguf(
                    prompt=prompt,
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    style=style,
                    target_language=target_language,
                    images=pil_images,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n_gpu_layers=n_gpu_layers,
                )
                return (enhanced_prompt,)
            except ValueError as e:
                # Translated
                error_str = str(e)
                if "Failed to load model" in error_str:
                    error_msg = (
                        f"ERROR: Failed to load model '{model}'\n"
                        "This is likely Qwen3-VL which requires llama-cpp-python >= 0.4.0\n"
                        "Current version: 0.3.16\n\n"
                        "Solutions:\n"
                        "1. Use Qwen2.5-VL instead (recommended)\n"
                        "2. Upgrade: pip install llama-cpp-python --upgrade"
                    )
                    print(f"[Vision LLM Node] {error_msg}")
                    raise RuntimeError(error_msg)
                else:
                    raise
        
        except RuntimeError:
            # Translated
            raise
        except Exception as e:
            print(f"[Vision LLM Node] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Unexpected error during prompt rewrite: {str(e)}")

# ============================================================================
# ComfyUI Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "VisionLLMNode": VisionLLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMNode": "Vision LLM Node"
}

# ============================================================================
# Cleanup on module unload
# ============================================================================

def cleanup():
    """Cleanup on module unload"""
    global _model_manager
    if _model_manager is not None:
        _model_manager.unload_model()

import atexit
atexit.register(cleanup)
