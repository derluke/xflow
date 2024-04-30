import logging
from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.framework.session import KedroBootSession
from joblib import Parallel, delayed

log = logging.getLogger(__name__)


class XFlowApp(AbstractKedroBootApp):
    def _run(self, kedro_boot_session: KedroBootSession):
        # leveraging config_loader to manage app's configs
        experiment_params = kedro_boot_session.config_loader["parameters"]
        log.info(f"experiment_params: {experiment_params}")
        experiments = kedro_boot_session.run(namespace="config")

        # log.info(f"experiments: {experiments}")
        # experiment_name = experiment_params["experiment_name"]
        def run_experiment(experiment):
            experiment_name = experiment["experiment_name"]
            log.info(f"Experiment_name: {experiment_name}")
            log.info("=" * 80)
            log.info(f"experiment: {experiment}")
            autopilot_run = kedro_boot_session.run(
                namespace="experiment",
                parameters={
                    "experiment_config": experiment,
                    "experiment_name": experiment_name,
                    "experiment_config.analyze_and_model.target": experiment[
                        "analyze_and_model"
                    ]["target"],
                    "experiment_config.binarize_data_config": experiment[
                        "binarize_data"
                    ],
                },
            )
            return {experiment_name: autopilot_run}

        results = Parallel(n_jobs=5, backend="threading")(
            delayed(run_experiment)(experiment) for experiment in experiments
        )
        log.info(f"results: {results}")
