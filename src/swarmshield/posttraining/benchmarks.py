import math
import re
from decimal import Decimal, InvalidOperation


def numeric_answer(text):
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def prompt_ids(tokenizer, text, device):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False,
    ).to(device)


def arc_prediction(model, tokenizer, row, device):
    import torch

    choices = row["choices"]
    options = "\n".join(f"{key}. {value}" for key, value in zip(choices["label"], choices["text"]))
    text = f"{row['question']}\n{options}\nReply with only the label of the correct answer."
    ids = prompt_ids(tokenizer, text, device)
    scores = []
    with torch.no_grad():
        for label in choices["label"]:
            answer = tokenizer.encode(label, add_special_tokens=False)
            tail = torch.tensor([answer], device=device)
            joined = torch.cat((ids, tail), dim=1)
            logits = model(input_ids=joined, attention_mask=torch.ones_like(joined), use_cache=False).logits
            relevant = logits[:, ids.shape[1] - 1 : joined.shape[1] - 1].float()
            score = relevant.log_softmax(-1).gather(-1, tail.unsqueeze(-1)).sum().item()
            if not math.isfinite(score):
                raise ValueError("Nonfinite ARC score")
            scores.append(score)
    prediction = choices["label"][max(range(len(scores)), key=scores.__getitem__)]
    return {"prediction": prediction, "reference": row["answerKey"], "correct": prediction == row["answerKey"], "scores": scores}


def gsm_prediction(model, tokenizer, row, device, max_new_tokens):
    import torch

    text = row["question"] + "\nSolve the problem. End with #### followed by the numeric answer."
    ids = prompt_ids(tokenizer, text, device)
    with torch.no_grad():
        result = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    tail = result[0, ids.shape[1]:]
    generated = tokenizer.decode(tail, skip_special_tokens=True)
    prediction, reference = numeric_answer(generated), numeric_answer(row["answer"])
    if reference is None:
        raise ValueError("Missing GSM reference answer")
    return {
        "prediction": prediction, "reference": reference, "correct": prediction == reference,
        "generated": generated, "generated_tokens": len(tail),
        "at_token_limit": len(tail) == max_new_tokens,
    }
