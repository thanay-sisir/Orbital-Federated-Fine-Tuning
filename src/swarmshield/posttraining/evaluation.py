def evaluate(model, examples, device):
    import math

    import torch

    model.eval()
    loss_sum, token_count = 0.0, 0
    with torch.no_grad():
        for row in examples:
            ids = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            loss = float(
                model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            )
            if not math.isfinite(loss):
                raise ValueError("Non-finite evaluation loss")
            n = sum(v != -100 for v in row["labels"][1:])
            loss_sum += loss * n
            token_count += n
    if token_count == 0:
        raise ValueError("Empty evaluation")
    return {
        "answer_loss": loss_sum / token_count,
        "supervised_tokens": token_count,
        "examples": len(examples),
        "task_quality": "not measured",
    }
