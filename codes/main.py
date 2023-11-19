from utils import  init_eos_experiment, init_trainer, setup_training, finish_run
import hydra
import logging

log = logging.getLogger(__name__)

@ hydra.main(config_path="../configs", config_name="base_train")
def main(config):
    setup_training(config)
    experiment = init_eos_experiment(config)
    trainer = init_trainer(experiment, config)
    if config.trainer.auto_lr_find:
        log.info(f"----- Tuning LR -----")
        trainer.tune(experiment)
        log.info(f"----- Completed LR Tuning -----")
    trainer.fit(experiment)
    run_api, _ = finish_run(trainer)


if __name__ == "__main__":
    main()
