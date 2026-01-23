# AGENTS.md

## Setup commands

- Activate the environment: `conda activate slam`
- Run the main script: `python slam_eval/scripts/main.py`
- Run tests: `pytest`
- Run linters: `./run_linters.sh`

## Main script configuration

- The main scripts is configured via hydra in `config/config_main.yaml`
- In general, all the user-specific fields (API tokens, paths etc.) should be stored in `config/user_settings/user_settings.yaml`. Do not commit `user_settings.yaml` without explicit permission

## Developer guide

### Basics

- Before making any changes, create a local git branch named `YYYYMMDD_short_task_description` where `YYYYMMDD` stands for the current date. Checkout this local branch and make all the changes there. Do not forget to make occasional commits during your work to be able to roll back to previous versions of the code if necessary. NEVER commit `config/user_settings/user_settings.yaml`, keep its changes unstaged.
- This code is configured using hydra. Its configs can be found in `config/`. No matter what constants/literals are used, they should be taken from configs. Object construction can also be made via `hydra.utils.instantiate` but use it only with simple classes (i.e., not derived from some base class).
- Run linters before finishing the job
- Use type hints, their use is necessiated by linters
- Update `README.md` when new functionality is added or there is outdated information

### How to add a new eval case collection for text generation

1. Derive a class from `EvalCaseCollection` and locate it in `slam_eval/collections/text_generation.py`
2. If necessary, add a custom scorer to `slam_eval/scorer.py` by deriving it from `Scorer`
3. Create a yaml config in `config/collection` for a new eval case collection and, if a new scorer is added, create its yaml config in `config/scorer`
4. To make the eval case collection ready for running, modify `collection` and `scorer` parameters in `config/config_main.yaml` corresponsingly
5. Since running `slam_eval/scripts/main.py` directly would imply making calls to an expensive LLM API, create tests in `tests/` instead ensuring that the new code is correct
