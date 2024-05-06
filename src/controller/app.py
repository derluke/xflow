import logging
import pickle

from joblib import Parallel, delayed
from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.framework.session import KedroBootSession

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
            output_dict = kedro_boot_session.run(
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
                    "experiment_config.group_data": experiment["group_data"],
                },
                itertime_params={"experiment_name": experiment_name},
            )
            return {
                "experiment_name": experiment_name,
                "project_id": output_dict,
                "experiment_config": experiment,
            }

        results = Parallel(n_jobs=10, backend="threading")(
            delayed(run_experiment)(experiment) for experiment in experiments
        )

        log.info(f"results: {results}")

        # with open("all_output.pickle", "wb") as f:
        #     pickle.dump(results, f)

        # with open("all_output.pickle", "rb") as f:
        #     results = pickle.load(f)
        # log.info(f"results: {results}")

        def run_measure(experiment):
            experiment_name = experiment["experiment_name"]
            log.info(f"Experiment_name: {experiment_name}")
            log.info("=" * 80)
            log.info(f"experiment: {experiment}")
            output_dict = kedro_boot_session.run(
                namespace="measure",
                parameters={
                    "experiment_name": experiment_name,
                    "experiment_config": experiment,
                    # "metrics": experiment["metrics"],
                },
                # inputs={"measure.project_dict": experiment["project_id"]},
                itertime_params={"experiment_name": experiment_name},
            )
            return {
                "experiment_name": experiment_name,
                "output_dict": output_dict,
                "experiment_config": experiment,
            }

        measure_results = Parallel(n_jobs=10, backend="threading")(
            delayed(run_measure)(experiment) for experiment in experiments
        )
        log.info(f"measure_results: {measure_results}")
        # from x_flow.pipelines.measure.nodes import select_candidate_models

        # candidate_models = select_candidate_models(results)
        # log.info(f"candidate_models: {candidate_models}")
