from transformers import AutoTokenizer

from still.chat import encode_system_prefix, encode_user_continuation
from still.config import DEFAULT_MATRIX
from still.eval.common import SYSTEM_PROMPT


def test_chat_split_reconstructs_full_prompt() -> None:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    system_prompt = SYSTEM_PROMPT.format(context="Example context.")
    user_message = "/no_think\nWhat is the answer?\n\nAnswer with one phrase."

    system_ids = encode_system_prefix(tokenizer, system_prompt)
    user_ids = encode_user_continuation(
        tokenizer,
        system_prompt=system_prompt,
        user_message=user_message,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    assert full_ids.shape[-1] == system_ids.shape[-1] + user_ids.shape[-1]
    assert full_ids.tolist()[0] == (system_ids.tolist()[0] + user_ids.tolist()[0])
