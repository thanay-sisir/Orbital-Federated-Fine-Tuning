import argparse
import json

from swarmshield.orbital.config import load, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orbital_sft/one_gpu_reference.yaml")
    args = parser.parse_args()
    base = load(args.config)
    output = project_path("configs/orbital_sft/generated")
    output.mkdir(parents=True, exist_ok=False)
    for seed in (42, 43, 44):
        for label, policy in (("fixed", "fixed"), ("aware", "contact_aware")):
            cfg = base.model_copy(deep=True)
            cfg.seed = seed
            cfg.network.policy = policy
            cfg.data.prepared_path = f"artifacts/prepared/seed{seed}-{cfg.data.partition}"
            (output / f"seed{seed}-{label}.yaml").write_text(
                json.dumps(cfg.model_dump(), indent=2) + "\n"
            )
    print(output)


if __name__ == "__main__":
    main()
