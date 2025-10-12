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
    Generate text and return both output and log probabilities.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        (generated_text, log_probs) where log_probs is list of log probabilities
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Generate with output scores
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Extract generated tokens (excluding input)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute log probabilities for generated tokens
    log_probs = []
    for i, score in enumerate(outputs.scores):
        # score shape: (batch_size, vocab_size)
        # Convert logits to log probabilities
        log_prob_dist = torch.log_softmax(score[0], dim=-1)

        # Get log prob of the actual generated token
        token_id = generated_ids[i]
        token_log_prob = log_prob_dist[token_id].item()
        log_probs.append(token_log_prob)

    return generated_text, log_probs
