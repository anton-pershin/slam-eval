from hydra.utils import instantiate
from omegaconf import DictConfig

import hydra
from slam_eval.utils.common import get_config_path

CONFIG_NAME = "config_main"


def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    collection = instantiate(cfg.collection)
    scorer = instantiate(cfg.scorer)
    eval_storage_adapter = instantiate(cfg.storage_adapter)

    collection.load()
    model_answers = []
    scores = []

    for eval_case in collection:

        y_pred = model.predict(eval_case["x"])
        score = scorer(eval_case["y_true"], y_pred)

        model_answers.append(y_pred)
        scores.append(score)

    eval_storage_adapter.save(
        group_id=cfg.group_id,
        model=model,
        eval_case_collection=collection,
        scores=scores,
        model_answers=model_answers,
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
