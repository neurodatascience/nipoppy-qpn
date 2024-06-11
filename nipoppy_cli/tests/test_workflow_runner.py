"""Tests for PipelineRunner."""

import json
from pathlib import Path

import pytest
from fids import fids

from nipoppy.config.main import Config
from nipoppy.tabular.bagel import Bagel
from nipoppy.tabular.doughnut import Doughnut
from nipoppy.utils import strip_session
from nipoppy.workflows.runner import PipelineRunner

from .conftest import create_empty_dataset, get_config


@pytest.fixture(scope="function")
def config(tmp_path: Path):
    fpath_descriptor = tmp_path / "descriptor.json"
    fpath_invocation = tmp_path / "invocation.json"

    descriptor = {
        "name": "dummy_pipeline",
        "tool-version": "1.0.0",
        "description": "A dummy pipeline for testing",
        "schema-version": "0.5",
        "command-line": "echo [ARG1] [ARG2] [[NIPOPPY_DPATH_BIDS]]",
        "inputs": [
            {
                "id": "arg1",
                "name": "arg1",
                "type": "String",
                "command-line-flag": "--arg1",
                "value-key": "[ARG1]",
            },
            {
                "id": "arg2",
                "name": "arg2",
                "type": "Number",
                "command-line-flag": "--arg2",
                "value-key": "[ARG2]",
            },
        ],
        "custom": {"nipoppy": {"CONTAINER_SUBCOMMAND": "exec"}},
    }
    invocation = {
        "arg1": "[[NIPOPPY_PARTICIPANT]] [[NIPOPPY_SESSION]]",
        "arg2": 10,
    }

    fpath_descriptor.write_text(json.dumps(descriptor))
    fpath_invocation.write_text(json.dumps(invocation))

    return get_config(
        visits=["BL", "V04"],
        container_config={"COMMAND": "echo"},  # dummy command
        proc_pipelines=[
            {
                "NAME": "dummy_pipeline",
                "VERSION": "1.0.0",
                "STEPS": [
                    {
                        "DESCRIPTOR_FILE": fpath_descriptor,
                        "INVOCATION_FILE": fpath_invocation,
                    }
                ],
            },
        ],
    )


def test_run_setup(config: Config, tmp_path: Path):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
    )
    create_empty_dataset(runner.dpath_root)
    config.save(runner.layout.fpath_config)
    runner.run_setup()
    assert runner.dpath_pipeline_output.exists()
    assert runner.dpath_pipeline_work.exists()


@pytest.mark.parametrize("simulate", [True, False])
def test_launch_boutiques_run(simulate, config: Config, tmp_path: Path):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
        simulate=simulate,
    )
    config.save(runner.layout.fpath_config)

    participant = "01"
    session = "ses-BL"

    fids.create_fake_bids_dataset(
        runner.layout.dpath_bids,
        subjects=participant,
        sessions=strip_session(session),
    )

    runner.dpath_pipeline_output.mkdir(parents=True, exist_ok=True)
    runner.dpath_pipeline_work.mkdir(parents=True, exist_ok=True)
    descriptor_str, invocation_str = runner.launch_boutiques_run(
        participant, session, container_command=""
    )

    assert "[[NIPOPPY_DPATH_BIDS]]" not in descriptor_str
    assert "[[NIPOPPY_PARTICIPANT]]" not in invocation_str
    assert "[[NIPOPPY_SESSION]]" not in invocation_str


def test_process_container_config_boutiques_subcommand(config: Config, tmp_path: Path):
    # check that the container subcommand from the Boutiques container config is used
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
    )

    config.save(runner.layout.fpath_config)

    participant = "01"
    session = "ses-BL"

    # the container command in the config is "echo"
    # because otherwise the check for the container command fails
    # if Singularity/Apptainer is not on the PATH
    assert (
        runner.process_container_config(participant=participant, session=session)
        == "echo exec"
    )


@pytest.mark.parametrize(
    "doughnut_data,bagel_data,pipeline_name,pipeline_version,expected",
    [
        (
            [
                ["01", "ses-1", False],
                ["01", "ses-2", True],
                ["01", "ses-3", True],
            ],
            None,
            "dummy_pipeline",
            "1.0.0",
            [("01", "ses-2"), ("01", "ses-3")],
        ),
        (
            [
                ["01", "ses-1", False],
                ["01", "ses-2", True],
                ["01", "ses-3", True],
            ],
            [],
            "dummy_pipeline",
            "1.0.0",
            [("01", "ses-2"), ("01", "ses-3")],
        ),
        (
            [
                ["01", "ses-1", False],
                ["01", "ses-2", True],
                ["01", "ses-3", True],
            ],
            [
                ["01", "ses-1", "dummy_pipeline", "1.0.0", Bagel.status_success],
                ["01", "ses-2", "dummy_pipeline", "1.0.0", Bagel.status_success],
                ["01", "ses-3", "dummy_pipeline", "1.0.0", Bagel.status_success],
            ],
            "dummy_pipeline",
            "1.0.0",
            [],
        ),
        (
            [
                ["01", "ses-1", True],
                ["01", "ses-2", True],
                ["01", "ses-3", True],
            ],
            [
                ["01", "ses-1", "dummy_pipeline", "1.0.0", Bagel.status_fail],
                ["01", "ses-2", "dummy_pipeline", "1.0.0", Bagel.status_success],
                ["01", "ses-3", "dummy_pipeline", "1.0.0", Bagel.status_fail],
                ["01", "ses-1", "dummy_pipeline", "2.0", Bagel.status_success],
            ],
            "dummy_pipeline",
            "1.0.0",
            [("01", "ses-1"), ("01", "ses-3")],
        ),
        (
            [
                ["01", "ses-1", True],
                ["01", "ses-2", True],
                ["01", "ses-3", True],
            ],
            [
                ["01", "ses-1", "dummy_pipeline", "1.0.0", Bagel.status_fail],
                ["01", "ses-2", "dummy_pipeline", "1.0.0", Bagel.status_success],
                ["01", "ses-3", "dummy_pipeline", "1.0.0", Bagel.status_fail],
                ["01", "ses-1", "dummy_pipeline", "2.0", Bagel.status_success],
            ],
            "dummy_pipeline",
            None,
            [("01", "ses-1"), ("01", "ses-3")],
        ),
    ],
)
def test_get_participants_sessions_to_run(
    doughnut_data,
    bagel_data,
    pipeline_name,
    pipeline_version,
    expected,
    config: Config,
    tmp_path: Path,
):
    participant = None
    session = None
    runner = PipelineRunner(
        dpath_root=tmp_path,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        participant=participant,
        session=session,
    )
    config.save(runner.layout.fpath_config)
    runner.doughnut = Doughnut().add_or_update_records(
        records=[
            {
                Doughnut.col_participant_id: data[0],
                Doughnut.col_session: data[1],
                Doughnut.col_visit: data[1],
                Doughnut.col_bidsified: data[2],
                Doughnut.col_datatype: None,
                Doughnut.col_participant_dicom_dir: "",
                Doughnut.col_dicom_id: "",
                Doughnut.col_bids_id: "",
                Doughnut.col_downloaded: False,
                Doughnut.col_organized: False,
            }
            for data in doughnut_data
        ]
    )
    if bagel_data is not None:
        Bagel(
            bagel_data,
            columns=[
                Bagel.col_participant_id,
                Bagel.col_session,
                Bagel.col_pipeline_name,
                Bagel.col_pipeline_version,
                Bagel.col_pipeline_complete,
            ],
        ).validate().save_with_backup(runner.layout.fpath_imaging_bagel)

    assert [
        tuple(x)
        for x in runner.get_participants_sessions_to_run(
            participant=participant, session=session
        )
    ] == expected
