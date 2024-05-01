"""
This module contains utility functions for working with DataRobot projects


"""

# pyright: reportPrivateImportUsage=false
import hashlib
import logging
import os
import tempfile
from collections import Counter
from typing import Any, List, Optional, Union

import datarobot as dr
import pandas as pd
from datarobot.utils import retry
from filelock import FileLock

log = logging.getLogger(__name__)


def wait_for_jobs(jobs: list[dr.Job]) -> None:
    """
    Wait for a list of jobs to complete
    """
    jobs = [j for j in jobs if j]
    for _, seconds_waited in retry.wait(24 * 60 * 60, maxdelay=20.0):
        # summarise the statuses by count of the jobs in the list
        job_status = Counter([j.status for j in jobs if j])
        # log.info the summary

        inprogress = job_status[
            dr.enums.QUEUE_STATUS.INPROGRESS  # pyright: ignore[reportAttributeAccessIssue]
        ]  #
        queued = job_status[
            dr.enums.QUEUE_STATUS.QUEUE  # pyright: ignore[reportAttributeAccessIssue]
        ]
        # completed = job_status[
        #     dr.enums.QUEUE_STATUS.COMPLETED  # pyright: ignore[reportAttributeAccessIssue]
        # ]

        log.info(
            f"Jobs: in progress: {inprogress}, "
            f"queued: {queued} (waited: {seconds_waited:.0f}s)"
        )
        # if all jobs are complete, break out of the loop
        if not (job_status["queue"] > 0 or job_status["inprogress"] > 0):
            break
        else:
            for j in jobs:
                j.refresh()


def get_models(project: dr.Project) -> Union[List[dr.Model], List[dr.DatetimeModel]]:
    if project.is_datetime_partitioned:
        return [
            m for m in project.get_datetime_models() if m.training_duration is not None
        ]
    else:
        return [m for m in project.get_models() if m.sample_pct is not None]


def _hash_pandas(df: pd.DataFrame) -> str:
    """
    Returns the hash of a pandas dataframe.
    :param self: pandas dataframe
    :return: hash of the pandas dataframe
    """
    return (
        str(
            int(
                hashlib.sha256(
                    pd.util.hash_pandas_object(df, index=True).values  # type: ignore
                ).hexdigest(),
                16,
            )
        )
        + ".csv.gz"
    )


def get_training_predictions(
    model: dr.Model, data_subset: dr.enums.DATA_SUBSET
) -> pd.DataFrame:
    """
    Get the training predictions for a model
    :param model: DataRobot model object
    :param data_subset: DataRobot data subset
    :return: pandas dataframe of training predictions
    """
    try:
        pred_job = model.request_training_predictions(data_subset=data_subset)
        tp = pred_job.get_result_when_complete()
    except dr.errors.ClientError as e:  # pylint: disable=broad-except
        # log.error(e)
        if (
            e.json["message"]
            == "`allBacktests` prediction set not valid if not all backtests have been computed"
        ):
            log.error(e)
            return None
        all_training_predictions = dr.TrainingPredictions.list(
            project_id=model.project_id
        )
        tp = [
            tp
            for tp in all_training_predictions
            if tp.model_id == model.id and tp.data_subset == data_subset
        ][0]
    return tp.get_all_as_dataframe()


def get_external_holdout_predictions(
    model: dr.Model,
    external_holdout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the external holdout predictions for a model
    :param model: DataRobot model object
    :param external_holdout: pandas dataframe of external holdout data
    :param partition_column: name of the partition column
    :return: pandas dataframe of external holdout predictions
    """
    project = dr.Project.get(model.project_id)  # type: ignore
    with FileLock(os.path.join(".locks", f"upload_dataset_{project.id}.lock")):
        ds = upload_dataset(project, external_holdout)
    pred_job = model.request_predictions(ds.id)
    pred_job.wait_for_completion()
    return pred_job.get_result_when_complete()  # type: ignore


def upload_dataset(
    project: dr.Project, sourcedata: pd.DataFrame, **kwags: Any
) -> dr.PredictionDataset:
    """
    Upload a pandas dataframe to a DataRobot project, or return the existing dataset if it has already been uploaded.

    :param project: DataRobot project object
    :param sourcedata: pandas dataframe to upload
    :param **kwags: additional keyword arguments to pass to project.upload_dataset
    :return: DataRobot dataset object

    """
    # check all uploaded dataset's filenames:
    datasets = project.get_datasets()
    uploaded_datasets = {d.name: d for d in datasets}

    hashed_df = _hash_pandas(sourcedata)
    if hashed_df in uploaded_datasets:
        return uploaded_datasets[hashed_df]
    else:
        # pop filename from kwargs
        _ = kwags.pop("dataset_filename", None)
        # compress data before uploading
        with tempfile.TemporaryDirectory() as tempdir:
            file_location = os.path.join(tempdir, hashed_df)
            sourcedata.to_csv(file_location, index=False, compression="gzip")
            dataset = project.upload_dataset(
                file_location, dataset_filename=hashed_df, **kwags
            )
        # project._uploaded_datasets[hashed_df] = dataset
        return dataset


# dr.Project.upload_dataset = upload_dataset


def predict(model: dr.Model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using a DataRobot model
    :param project: DataRobot project object
    :param df: pandas dataframe to make predictions on
    :return: pandas dataframe of predictions
    """
    project = dr.Project.get(model.project_id)

    ds = upload_dataset(project, df)
    pred_job = model.request_predictions(ds.id)
    pred_job.wait_for_completion()
    return pred_job.get_result_when_complete()  # type: ignore


def calculate_stats(  # noqa: PLR0912,PLR0915
    project: dr.Project, models: int = 5, verbose: bool = False
) -> None:
    """
    Calculate stats for the first n_models models in the project
    :param project: project object
    :param n_models: int number of models to calculate stats for
    :return: None
    """
    project.set_worker_count(-1)
    assert project.id

    def wait_for_jobs(jobs: list[dr.Job]) -> None:
        """
        Wait for a list of jobs to complete
        """
        jobs = [j for j in jobs if j]
        for _, seconds_waited in retry.wait(24 * 60 * 60, maxdelay=20.0):
            # summarise the statuses by count of the jobs in the list
            job_status = Counter([j.status for j in jobs if j])
            # log.info the summary

            inprogress = job_status[
                dr.enums.QUEUE_STATUS.INPROGRESS  # pyright: ignore[reportAttributeAccessIssue]
            ]  #
            queued = job_status[
                dr.enums.QUEUE_STATUS.QUEUE  # pyright: ignore[reportAttributeAccessIssue]
            ]
            # completed = job_status[
            #     dr.enums.QUEUE_STATUS.COMPLETED  # pyright: ignore[reportAttributeAccessIssue]
            # ]

            log.info(
                f"Insights: in progress: {inprogress}, "
                f"queued: {queued} (waited: {seconds_waited:.0f}s)"
            )
            # if all jobs are complete, break out of the loop
            if not (job_status["queue"] > 0 or job_status["inprogress"] > 0):
                break
            else:
                for j in jobs:
                    j.refresh()

    if isinstance(models, int):
        if project.is_datetime_partitioned:
            models = project.get_datetime_models()[:models]
        else:
            models = [m for m in project.get_models()[:models] if m]  # type: ignore

    def score_backtests(m: dr.DatetimeModel) -> dr.Job:
        try:
            return m.score_backtests()
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    def calculate_shap_impact(m: dr.Model) -> dr.Job:
        try:
            return dr.ShapImpact.create(project_id=m.project_id, model_id=m.id)
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    def cross_validate(m: dr.Model) -> dr.Job:
        try:
            return m.cross_validate()
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    def request_feature_impact(m: dr.Model) -> dr.Job:
        try:
            return m.request_feature_impact()
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    def request_feature_effect(m: dr.Model, backtest: Optional[str] = None) -> dr.Job:
        try:
            if backtest is None:
                return m.request_feature_effect()
            else:
                return m.request_feature_effect(backtest)
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    def compute_datetime_trend_plots(
        m: dr.DatetimeModel, backtest: Union[str, int], source: Optional[str]
    ) -> dr.Job:
        try:
            return m.compute_datetime_trend_plots(backtest, source)
        except Exception as e:  # pylint: disable=broad-except
            if verbose:
                log.info(e)
            return None

    # calculate FI for all models
    jobs = []
    for m in models:
        jobs.append(request_feature_impact(m))
    wait_for_jobs(jobs)

    jobs = []
    for m in models:
        jobs.append(calculate_shap_impact(m))
    wait_for_jobs(jobs)

    if project.is_datetime_partitioned:
        dtp = dr.DatetimePartitioning.get(project.id)
        jobs = []
        jobs += [score_backtests(m) for m in models]
        wait_for_jobs(jobs)

        jobs = []
        models = [dr.DatetimeModel.get(project.id, m.id) for m in models]  # type: ignore
        for m in models:
            for i in list(range(dtp.number_of_backtests)) + [
                dr.enums.DATA_SUBSET.HOLDOUT
            ]:
                jobs.append(request_feature_effect(m, str(i)))
                if project.use_time_series:
                    for source in ["training", "validation"]:
                        try:
                            jobs.append(
                                compute_datetime_trend_plots(
                                    m, backtest=str(i), source=source
                                )
                            )
                        except Exception as e:  # pylint: disable=broad-except
                            if verbose:
                                log.info(f"{m.id}, {i}, {source}, failed, {e}")
    else:
        jobs = []
        for m in models:
            jobs.append(cross_validate(m))
            jobs.append(request_feature_effect(m))

    wait_for_jobs(jobs)
