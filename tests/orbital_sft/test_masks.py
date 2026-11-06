import pytest

from swarmshield.posttraining.data import encode_conversation


class CharacterTemplate:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>" + m["content"] + "</turn>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


@pytest.mark.parametrize("turns", [1, 2])
def test_only_assistant_spans_supervised(turns):
    messages = [{"role": "system", "content": "SYSTEM"}]
    for _ in range(turns):
        messages.extend(
            [{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "ANSWER"}]
        )
    encoded, _dropped = encode_conversation(CharacterTemplate(), messages, 1000)
    text = "".join(chr(i) for i in encoded["input_ids"])
    learned = "".join(chr(i) for i in encoded["labels"] if i != -100)
    assert "PROMPT" in text and "SYSTEM" not in learned and "PROMPT" not in learned
    assert learned.count("ANSWER") == turns


def test_overlength_counts_lost_targets():
    encoded, dropped = encode_conversation(
        CharacterTemplate(),
        [{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "ANSWER"}],
        16,
    )
    assert encoded is None and dropped["supervised_tokens"] > 0


def test_invalid_roles_rejected():
    with pytest.raises(ValueError):
        encode_conversation(CharacterTemplate(), [{"role": "assistant", "content": "answer"}], 100)
