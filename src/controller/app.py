import logging
import pickle
from typing import Any

from joblib import Parallel, delayed
from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.framework.session import KedroBootSession

log = logging.getLogger(__name__)

# TODO: datarobot version stuff


class XFlowApp(AbstractKedroBootApp):  # type: ignore
    """Main application class for the XFlow project."""

    def _run(self, kedro_boot_session: KedroBootSession) -> None:
        def log_experiment_info(experiment_name: str, experiment: dict[str, Any]) -> None:
            log.info(f"Experiment_name: {experiment_name}")
            log.info("=" * 80)
            log.info(f"experiment: {experiment}")

        def get_parameters(experiment: dict[str, Any], namespace: str) -> dict[str, Any]:
            if namespace == "experiment":
                return {
                    "experiment_config": experiment,
                    "experiment_name": experiment["experiment_name"],
                    "experiment_config.analyze_and_model.target": experiment["analyze_and_model"][
                        "target"
                    ],
                    "experiment_config.binarize_data_config": experiment.get("binarize_data", {}),
                    "experiment_config.partition_column": experiment.get("group_data", {}).get(
                        "partition_column", None
                    ),
                }
            elif namespace == "measure":
                return {
                    "experiment_name": experiment["experiment_name"],
                    "experiment_config": experiment,
                }
            else:
                raise ValueError(f"Unknown namespace: {namespace}")

        def run_namespace_session(experiment: dict[str, Any], namespace: str) -> dict[str, Any]:
            experiment_name = experiment["experiment_name"]
            log_experiment_info(experiment_name, experiment)
            output_dict = kedro_boot_session.run(
                namespace=namespace,
                parameters=get_parameters(experiment, namespace),
                itertime_params={"experiment_name": experiment_name},
            )
            return {
                "experiment_name": experiment_name,
                "output_dict": output_dict,
                "experiment_config": experiment,
            }

        def run_experiments(
            experiments: list[dict[str, Any]], namespace: str
        ) -> list[dict[str, Any]]:
            results = Parallel(n_jobs=100, backend="threading")(
                delayed(run_namespace_session)(experiment, namespace) for experiment in experiments
            )
            # log.info(f"{namespace}_results: {results}")
            return list(results)

        # leveraging config_loader to manage app's configs
        experiments = kedro_boot_session.run(namespace="config")

        run_experiments(experiments, "experiment")
        measure_results = run_experiments(experiments, "measure")

        # save results
        # with open("experiment_results.pkl", "wb") as f:
        #     pickle.dump(experiment_results, f)

        with open("measure_results.pkl", "wb") as f:
            pickle.dump(measure_results, f)
