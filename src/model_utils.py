"""
Model loading utilities with quantization and LoRA support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_model_optimized(model_name: str):
    """
    Load model with appropriate quantization based on size.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine loading strategy based on model size
    model_name_lower = model_name.lower()

    if "70b" in model_name_lower or "72b" in model_name_lower:
        # Use 4-bit quantization for 70B models
        print("  Using 4-bit quantization for large model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    elif any(x in model_name_lower for x in ["nemo", "12b", "13b"]):
        # Use 8-bit for 12B+ models
        print("  Using 8-bit quantization for medium model")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        # Full precision (float16) for smaller models
        print("  Using float16 for smaller model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Add LoRA adapters (trainable even with quantization)
    print("  Adding LoRA adapters")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Enabled gradient checkpointing")

    # Enable input gradients for LoRA
    model.enable_input_require_grads()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def generate_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
):
    """
    Generate text with memory-efficient sampling that maintains gradients.

    Memory optimizations:
    - Uses KV cache to avoid recomputing past tokens
    - Only keeps gradients for sampled log probs
    - Clears cache after generation
    - Detaches token IDs (no gradient needed)

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        (generated_text, log_probs) where log_probs is list of tensors with gradients
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    generated_tokens = []
    log_probs = []
    past_key_values = None

    # First forward pass processes entire prompt (keep gradients for sampling)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :] / temperature

    # Sample first token with gradients
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    token_log_prob = torch.log(probs[0, next_token[0]])
    log_probs.append(token_log_prob)
    generated_tokens.append(next_token[0].item())

    if next_token[0].item() == tokenizer.eos_token_id:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Clear KV cache
        del past_key_values
        torch.cuda.empty_cache()
        return generated_text, log_probs

    # Update attention mask
    attention_mask = torch.cat([
        attention_mask,
        torch.ones((1, 1), device=model.device, dtype=attention_mask.dtype)
    ], dim=1)

    # Continue generation with KV cache
    current_token = next_token.detach()  # Detach - no gradient needed for token IDs

    for _ in range(max_new_tokens - 1):
        # Forward pass with KV cache - only process new token
        outputs = model(
            input_ids=current_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Update cache
        past_key_values = outputs.past_key_values

        # Get logits for next token
        next_token_logits = outputs.logits[:, -1, :] / temperature

        # Sample next token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Get log prob of sampled token (keep gradient)
        token_log_prob = torch.log(probs[0, next_token[0]])
        log_probs.append(token_log_prob)

        # Store generated token
        generated_tokens.append(next_token[0].item())

        # Stop if EOS token
        if next_token[0].item() == tokenizer.eos_token_id:
            break

        # Update for next iteration
        current_token = next_token.detach()  # Detach token IDs
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=model.device, dtype=attention_mask.dtype)
        ], dim=1)

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clear KV cache to free memory
    del past_key_values
    del outputs
    torch.cuda.empty_cache()

    return generated_text, log_probs
