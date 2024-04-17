import logging
from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.framework.session import KedroBootSession

log = logging.getLogger(__name__)


class XFlowApp(AbstractKedroBootApp):
    def _run(self, kedro_boot_session: KedroBootSession):
        # leveraging config_loader to manage app's configs
        experiment_params = kedro_boot_session.config_loader["parameters"]

        experiments = kedro_boot_session.run(namespace="config")
        # log.info(f"experiments: {experiments}")
        for experiment in experiments:
            log.info(f"experiment: {experiment}")
            # kedro_boot_session.run(
            #     namespace="experiment",
            #     inputs={"experiment": experiment},
            #     parameters={"experiment_params": experiment_params},
            # )
