import torch


def chat_template_kwargs() -> dict[str, bool]:
    """Return the chat-template flags shared by all benchmark prompt builders."""
    return {"enable_thinking": False}


def encode_system_prefix(tokenizer, system_prompt: str) -> torch.Tensor:
    """Encode just the system portion of the chat template for cache construction."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False,
        add_generation_prompt=False,
        **chat_template_kwargs(),
    )
    return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def encode_user_continuation(
    tokenizer,
    *,
    system_prompt: str,
    user_message: str,
    assistant_prefix: str = "",
) -> torch.Tensor:
    """Encode the user continuation after verifying the system-prefix split matches exactly."""
    system_ids = encode_system_prefix(tokenizer, system_prompt)
    full_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(),
    )
    full_ids = tokenizer(
        full_prompt + assistant_prefix, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    prefix_len = int(system_ids.shape[-1])
    if prefix_len >= int(full_ids.shape[-1]):
        raise ValueError("System prefix consumed the entire prompt; expected a user continuation.")
    if not torch.equal(full_ids[:, :prefix_len], system_ids):
        raise ValueError(
            "Chat-template system prefix mismatch while splitting system and user tokens."
        )
    return full_ids[:, prefix_len:]
