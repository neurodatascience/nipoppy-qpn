"""Tests for the PipelineTracker class."""

from pathlib import Path

import pytest

from nipoppy.config.main import Config
from nipoppy.tabular.bagel import Bagel
from nipoppy.tabular.manifest import Manifest
from nipoppy.workflows.tracker import PipelineTracker

from .conftest import create_empty_dataset, get_config, prepare_dataset


@pytest.fixture(scope="function")
def tracker(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    pipeline_name = "test_pipeline"
    pipeline_version = "0.1.0"
    participants_and_sessions = {
        "01": ["ses-1", "ses-2"],
        "02": ["ses-1", "ses-2"],
    }

    tracker = PipelineTracker(
        dpath_root=dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
    )

    create_empty_dataset(dpath_root)

    manifest: Manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
    )
    manifest.save_with_backup(tracker.layout.fpath_manifest)

    config: Config = get_config(
        visits=["1", "2"],
        proc_pipelines={
            pipeline_name: {
                pipeline_version: {
                    "TRACKER_CONFIG": {
                        "pipeline_complete": [
                            "[[NIPOPPY_PARTICIPANT]]/[[NIPOPPY_SESSION]]/results.txt",
                            "file.txt",
                        ]
                    }
                }
            }
        },
    )
    config.save(tracker.layout.fpath_config)

    return tracker


def test_run_setup(tracker: PipelineTracker):
    tracker.run_setup()
    assert tracker.bagel.empty


def test_run_setup_existing_bagel(tracker: PipelineTracker):
    bagel = Bagel(
        data={
            Bagel.col_participant_id: ["01"],
            Bagel.col_session: ["ses-1"],
            Bagel.col_pipeline_name: ["some_pipeline"],
            Bagel.col_pipeline_version: ["some_version"],
            Bagel.col_pipeline_complete: [Bagel.status_success],
        }
    ).validate()
    bagel.save_with_backup(tracker.layout.fpath_imaging_bagel)

    tracker.run_setup()

    assert tracker.bagel.equals(bagel)


@pytest.mark.parametrize(
    "relative_paths,expected_status",
    [
        (["01_ses-1.txt", "file.txt"], Bagel.status_success),
        (["01_ses-1.txt", "file.txt", "missing.txt"], Bagel.status_fail),
    ],
)
def test_check_status(tracker: PipelineTracker, relative_paths, expected_status):
    for relative_path_to_write in ["01_ses-1.txt", "file.txt"]:
        fpath = tracker.dpath_pipeline_output / relative_path_to_write
        fpath.mkdir(parents=True, exist_ok=True)
        fpath.touch()

    assert tracker.check_status(relative_paths) == expected_status


@pytest.mark.parametrize(
    "participant,session,expected_status",
    [("01", "ses-1", Bagel.status_success), ("02", "ses-2", Bagel.status_fail)],
)
def test_run_single(participant, session, expected_status, tracker: PipelineTracker):
    for relative_path_to_write in [
        "01/ses-1/results.txt",
        "file.txt",
        "02/ses-1/results.txt",
    ]:
        fpath = tracker.dpath_pipeline_output / relative_path_to_write
        fpath.mkdir(parents=True, exist_ok=True)
        fpath.touch()

    assert tracker.run_single(participant, session) == expected_status

    assert (
        tracker.bagel.set_index([Bagel.col_participant_id, Bagel.col_session])
        .loc[:, Bagel.col_pipeline_complete]
        .item()
    ) == expected_status


@pytest.mark.parametrize(
    "bagel",
    [
        Bagel(),
        Bagel(
            data={
                Bagel.col_participant_id: ["01"],
                Bagel.col_session: ["ses-1"],
                Bagel.col_pipeline_name: ["some_pipeline"],
                Bagel.col_pipeline_version: ["some_version"],
                Bagel.col_pipeline_complete: [Bagel.status_success],
            }
        ).validate(),
    ],
)
def test_run_cleanup(tracker: PipelineTracker, bagel: Bagel):
    tracker.bagel = bagel
    tracker.run_cleanup()

    assert tracker.layout.fpath_imaging_bagel.exists()
    assert Bagel.load(tracker.layout.fpath_imaging_bagel).equals(bagel)
