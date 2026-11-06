import hashlib

from swarmshield.orbital.config import project_path


def frozen_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if "lora_" not in name:
            h.update(name.encode())
            h.update(
                tensor.detach()
                .cpu()
                .contiguous()
                .view(-1)
                .view(__import__("torch").uint8)
                .numpy()
                .tobytes()
            )
    return h.hexdigest()


def load_model(cfg, device):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA GPU")
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.training_seed)
    torch.cuda.manual_seed_all(cfg.training_seed)
    loader = getattr(transformers, cfg.model.loader, None)
    if loader is None:
        raise ValueError("Configured architecture loader is absent from this Transformers build")
    model = loader.from_pretrained(
        project_path(cfg.model.path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    )
    modules = dict(model.named_modules())
    for target in cfg.model.target_modules:
        if target not in modules or any(
            word in target.lower() for word in ("visual", "vision", "merger")
        ):
            raise ValueError(f"Missing or non-language target: {target}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    pattern = "^(?:" + "|".join(__import__("re").escape(x) for x in cfg.model.target_modules) + ")$"
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.model.rank,
            lora_alpha=cfg.model.alpha,
            target_modules=pattern,
            lora_dropout=0.0,
            bias="none",
        ),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_B" in name)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("No B factors were attached")
    model.config.use_cache = False
    return model


def adapter_state(model):
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.named_parameters()
        if "lora_" in name
    }


def install(model, state):
    import torch

    actual = {name: tensor for name, tensor in model.named_parameters() if "lora_" in name}
    if actual.keys() != state.keys():
        raise ValueError("Adapter key mismatch")
    with torch.no_grad():
        for name, tensor in actual.items():
            if tensor.shape != state[name].shape or not torch.isfinite(state[name]).all():
                raise ValueError("Adapter shape or finite-value mismatch")
            tensor.copy_(state[name].to(tensor.device, tensor.dtype))


def fixed_a_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        if "lora_A" in name:
            h.update(name.encode())
            h.update(state[name].float().contiguous().numpy().tobytes())
    return h.hexdigest()
