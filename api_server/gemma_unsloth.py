import logging
from unsloth import FastModel
import torch
import numpy as np
import librosa
import base64
import io
from PIL import Image

# increase cache size limit
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
# /home/dev/.local/.unsloth_gemma-3n-E2B-it-unsloth-bnb-4bit # 9 GB 
# bad at unserstand arabic people speak english

# /home/dev/.local/.unsloth_gemma-3n-E4B-it-unsloth-bnb-4bit # 11 GB
# Good at unserstand arabic people speak english

MAX_INPUT_TOKENS = 2048-512-256  # Limit input context
MAX_TOTAL_TOKENS = 2048-512  # Total context including response
# MAX_INPUT_TOKENS = 2048  # Limit input context
# MAX_TOTAL_TOKENS = 2048 + 512  # Total context including response
VRAM_LIMIT_GB = 11.5

def check_vram():
    allocated = torch.cuda.memory_allocated() / 1024**3
    if allocated > VRAM_LIMIT_GB:
        torch.cuda.empty_cache()
        return False
    return True

model = None
tokenizer = None
current_model_name = None

MODELS = {
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit": "/home/dev/.local/.unsloth_gemma-3n-E2B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit": "/home/dev/.local/.unsloth_gemma-3n-E4B-it-unsloth-bnb-4bit",
}

def load_model(model_name: str):
    global model, tokenizer, current_model_name
    if model_name == current_model_name and model is not None and tokenizer is not None:
        return model, tokenizer

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()

    model_path = MODELS.get(model_name)
    if not model_path:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}")

    logging.info(f"Loading model: {model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        max_seq_length=MAX_TOTAL_TOKENS,
        load_in_4bit=True,
        full_finetuning=False,
    )
    current_model_name = model_name
    logging.info(f"Model {model_name} loaded successfully.")
    return model, tokenizer

# Let's first experience how Gemma 3N can handle multimodal inputs. We use Gemma 3N's recommended settings of temperature = 1.0, top_p = 0.95, top_k = 64
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())

from transformers import TextStreamer
# Helper function for inference
def load_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def print_memory_stats(label="", seq_length=None):
    print(f"\n=== Memory Stats {label} ===")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    if seq_length is not None:
        # KV cache size = num_layers * 2 * batch_size * seq_len * hidden_dim * bytes_per_element
        kv_size = 28 * 2 * 1 * seq_length[-1] * 2560 * 0.5 / 1024**3  # For 4-bit
        print(f"Estimated KV Cache: {kv_size:.2f} GB")

def do_gemma_3n_inference(messages, model_name: str, max_new_tokens=512):
    try:
        model, tokenizer = load_model(model_name)
        if not check_vram():
            logging.error("VRAM limit exceeded during inference check")
            raise Exception("VRAM limit exceeded")

        # Process messages to convert base64 images and audio to PIL Images and audio arrays
        for msg in messages:
            if not isinstance(msg.get('content'), list):
                continue

            processed_content_parts = []
            for content_item in msg['content']:
                if content_item.get('type') == 'image_url':
                    image_url = content_item.get('image_url', {}).get('url')
                    if image_url and image_url.startswith('data:image'):
                        try:
                            header, encoded = image_url.split(',', 1)
                            image_data = base64.b64decode(encoded)
                            image = Image.open(io.BytesIO(image_data))
                            processed_content_parts.append({"type": "image", "image": image})
                        except Exception as e:
                            logging.error(f"Error decoding image: {e}")
                            processed_content_parts.append(content_item)
                elif content_item.get('type') == 'audio_url':
                    audio_url = content_item.get('audio_url', {}).get('url')
                    if audio_url and audio_url.startswith('data:audio'):
                        try:
                            header, encoded = audio_url.split(',', 1)
                            audio_data = base64.b64decode(encoded)
                            audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
                            processed_content_parts.append({"type": "audio", "audio": audio})
                        except Exception as e:
                            logging.error(f"Error decoding audio: {e}")
                            processed_content_parts.append(content_item)
                else:
                    processed_content_parts.append(content_item)
            
            msg['content'] = processed_content_parts

        logging.info("\n=== Model Input ===")
        logging.info(f"Messages: {messages}")
        
        # Create attention mask for input
        try:
            inputs = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            logging.error(f"Error during tokenizer.apply_chat_template: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return None
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(inputs["input_ids"])
        inputs["attention_mask"] = attention_mask
        
        logging.info(f"Tokenized input shape: {inputs['input_ids'].shape}")
        
        # Truncate inputs if they are too long to prevent out of memory errors.
        if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            logging.warning(f"Input length {inputs['input_ids'].shape[1]} exceeds MAX_INPUT_TOKENS {MAX_INPUT_TOKENS}. Truncating.")
            inputs["input_ids"] = inputs["input_ids"][:, -MAX_INPUT_TOKENS:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_INPUT_TOKENS:]
        
        try:
            outputs = model.generate(
                **inputs.to("cuda"),
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        except Exception as e:
            logging.error(f"Error during model.generate: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return None

        try:
            # Find the starting position of the generated text
            start_pos = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            # Decode the generated text, skipping special tokens
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the newly generated text
            if len(decoded_text) > start_pos:
                generated_text = decoded_text[start_pos:]
            else:
                generated_text = ""
        except IndexError:
            # Handle cases where the output is empty or malformed
            generated_text = ""

        logging.info("\n=== Model Output ===")
        logging.info(f"Raw output: {generated_text}")
        
        torch.cuda.empty_cache()
        return generated_text

    except Exception as e:
        logging.error(f"Error during inference: {e}", exc_info=True)
        torch.cuda.empty_cache()
        return None
