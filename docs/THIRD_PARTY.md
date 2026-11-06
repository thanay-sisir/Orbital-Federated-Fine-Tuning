# Model, data and research references

Model weights and raw training data are downloaded separately and are not redistributed in this repository. Their use remains subject to their publisher terms. The source-selection review is in [DATA_TERMS.md](orbital_sft/DATA_TERMS.md).

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), pinned in the experiment configurations.
- Data collection: [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), restricted to its OpenAssistant source.
- OpenAssistant: Kopf et al., *OpenAssistant Conversations — Democratizing Large Language Model Alignment* (2023).
- Tulu: Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training* (2024).
- IFEval: [Google Research instruction-following evaluation](https://github.com/google-research/google-research/tree/master/instruction_following_eval). The recorded study used pinned official checkers; upstream notices and terms apply when staging them.
- Numerical geometry: [Skyfield](https://rhodesmill.org/skyfield/earth-satellites.html), [SGP4](https://pypi.org/project/sgp4/), and [Astropy satellite coordinate guidance](https://docs.astropy.org/en/stable/coordinates/satellites.html).

Dependencies keep their own licenses. External tools and research references do not imply agency endorsement, flight qualification or independent validation of this project. The four public graphics are reproducible vector/data visualizations generated from this project's recorded evidence, with metric inputs and scripts included.
