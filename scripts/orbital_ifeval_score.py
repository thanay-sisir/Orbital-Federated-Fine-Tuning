import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def scorer():
    manifest = json.loads((ROOT / "artifacts/benchmarks/ifeval-v1/manifest.json").read_text())
    source = ROOT / manifest["scorer_root"]
    for name, expected in manifest["scorer_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("IFEval scorer checksum mismatch")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
    if hashlib.sha256(freeze).hexdigest() != manifest["scoring_environment_freeze_sha256"]:
        raise ValueError("IFEval scoring environment differs")
    for name, expected in manifest["nltk_resource_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("IFEval NLTK resource checksum mismatch")
    os.environ["NLTK_DATA"] = str(ROOT / ".cache/nltk_data")
    sys.path.insert(0, str(source))
    import langdetect
    from instruction_following_eval import evaluation_lib

    langdetect.DetectorFactory.seed = 42
    random.seed(42)
    return evaluation_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses")
    parser.add_argument("--output")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    lib = scorer()
    if args.self_check:
        item = lib.InputExample(1, ["detectable_format:json_format"], "Reply in JSON", [{}])
        assert lib.test_instruction_following_strict(item, {item.prompt: '{"answer": "valid"}'}).follow_all_instructions
        assert not lib.test_instruction_following_strict(item, {item.prompt: "not valid JSON"}).follow_all_instructions
        assert not lib.test_instruction_following_loose(item, {item.prompt: ""}).follow_all_instructions
        for example in lib.read_prompt_list(str(ROOT / "artifacts/benchmarks/ifeval-v1/prompts.jsonl")):
            response = {example.prompt: "This is a short response. It has two sentences."}
            lib.test_instruction_following_strict(example, response)
            lib.test_instruction_following_loose(example, response)
        print("Official scorer positive/negative checks passed")
        return
    prompts = ROOT / "artifacts/benchmarks/ifeval-v1/prompts.jsonl"
    inputs = lib.read_prompt_list(str(prompts))
    responses = lib.read_prompt_to_response_dict(args.responses)
    if set(responses) != {item.prompt for item in inputs}:
        raise ValueError("IFEval responses missing or unexpected")
    output = {}
    for mode, evaluate in (("strict", lib.test_instruction_following_strict), ("loose", lib.test_instruction_following_loose)):
        values = [evaluate(item, responses) for item in inputs]
        flags = [flag for result in values for flag in result.follow_instruction_list]
        output[mode] = {
            "prompt_accuracy": sum(item.follow_all_instructions for item in values) / len(values),
            "instruction_accuracy": sum(flags) / len(flags),
            "prompts": len(values), "instructions": len(flags),
        }
        details = Path(args.output).with_suffix(f".{mode}.jsonl")
        with details.open("x") as stream:
            for result in values:
                stream.write(json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n")
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
