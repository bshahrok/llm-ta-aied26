"""
Model loading and inference management.
"""

import json
import logging
from typing import Callable, Optional, Dict
import re 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CPU_MODEL_ID, GPU_MODEL_ID, MAX_NEW_TOKENS
from utils.memory import clear_cache, log_memory_usage, setup_memory_optimization

logger = logging.getLogger(__name__)


class MockModelManager:
    """Lightweight mock LLM for tests and dry runs."""

    def __init__(
        self,
        response: Optional[str] = None,
        generator: Optional[Callable[[str], str]] = None,
    ):
        self.response = response
        self.generator = generator
        self.model_id = "mock"
        self.device = torch.device("cpu")
        logging.info("Initialized MockModelManager")

    def load_model(self):
        """No-op for mock model."""
        return
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a deterministic mock response."""
        if self.generator:
            return self.generator(prompt)
        if self.response:
            return self.response
        return json.dumps({
            "CAD-code": "NONE",
            "rationale": "Mock LLM response."
        })

    def unload_model(self):
        """No-op for mock model."""
        return


class ModelManager:
    """Handles model loading and inference."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        use_8bit: bool = False,
        use_4bit: bool = True,  # Essential for 7B on 8GB GPU
        max_memory_gb: Optional[float] = 7.5  # Leave some headroom
    ):
        self.device = self._resolve_device(device)
        self.model_id = model_id or self._get_default_model_id()
        self.temperature = temperature
        self.top_k = top_k
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.max_memory_gb = max_memory_gb

        self._tokenizer = None
        self._model = None
        # Setup memory optimization
        setup_memory_optimization()

    def _resolve_device(self, device: Optional[str]):
        """Determines the appropriate device for model execution."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_default_model_id(self) -> str:
        """Selects default model based on available hardware."""
        if self.device.type == "cuda":
            return GPU_MODEL_ID
        return CPU_MODEL_ID

    def load_model(self):
        """Loads tokenizer and model if not already loaded."""
        if self._model and self._tokenizer:
            return
        
        logger.info(f"Loading model: {self.model_id}")
        clear_cache()
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Build model loading kwargs
            model_kwargs = {"device_map": "auto" if self.device.type == "cuda" else "cpu"}

            # Add quantization for GPU
            if self.device.type == "cuda":
                if self.use_4bit or self.use_8bit:
                    try:
                        from transformers import BitsAndBytesConfig
                    except ImportError:
                        logger.warning("bitsandbytes not installed; disabling quantization")
                        self.use_4bit = False
                        self.use_8bit = False
                    else:
                        if self.use_4bit:
                            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                            logger.info("Loading with 4-bit quantization")
                        elif self.use_8bit:
                            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True
                            )
                            logger.info("Loading with 8-bit quantization")

                if "quantization_config" not in model_kwargs:
                    # Only set dtype if NOT using quantization
                    model_kwargs["torch_dtype"] = torch.float16
                
                if self.max_memory_gb and self.device.type == "cuda":
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        model_kwargs["max_memory"] = {
                            i: f"{self.max_memory_gb}GB" for i in range(device_count)
                        }

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            self._model.eval()

            # Log memory after loading
            logger.info("Model loaded successfully")
            log_memory_usage()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM while loading model: {e}")
            clear_cache()
            raise RuntimeError(
                "Out of GPU memory. Try: \n"
                "1. Use smaller model (1.5B instead of 7B)\n"
                "2. Enable quantization: use_8bit=True or use_4bit=True\n"
                "3. Set max_memory_gb to limit memory per GPU\n"
                "4. Close other GPU processes"
            ) from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generates text from the model given a prompt."""
        self.load_model()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model or tokenizer is not loaded.")
        temp = temperature if temperature is not None else self.temperature
        tk = top_k if top_k is not None else self.top_k

        try:
            # Clear cache before generation
            clear_cache()

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(self._model.device)

            generate_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token_id": self._tokenizer.eos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            
            do_sample = float(temp) > 0
            generate_kwargs.update({
                "do_sample": do_sample,
                "temperature": float(temp),
            })
            if do_sample:
                generate_kwargs["top_k"] = int(tk)

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generate_kwargs)
            
            full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_text = self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            
            # Clean up inputs/outputs tensors
            del inputs, outputs
            clear_cache()

            if full_text.startswith(prompt_text):
                generated = full_text[len(prompt_text):].strip()
            else:
                generated = full_text.strip()

            return generated

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during generation: {e}")
            log_memory_usage()
            clear_cache()
            raise RuntimeError(
                "Out of GPU memory during generation. Try:\n"
                "1. Reduce max_new_tokens\n"
                "2. Process texts in smaller batches\n"
                "3. Unload and reload model: agent.model_manager.unload_model()\n"
                "4. Enable quantization if not already enabled"
            ) from e

    def unload_model(self):
        """Unload model from memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        clear_cache()
        logger.info("Model unloaded from memory")