"""DICOM file organization."""

import logging
import os
from pathlib import Path
from typing import Optional

from nipoppy.utils import StrOrPathLike
from nipoppy.workflows.base import BaseWorkflow


class DicomReorgWorkflow(BaseWorkflow):
    """Workflow for organizing raw DICOM files."""

    def __init__(
        self,
        dpath_root: StrOrPathLike,
        copy_files: bool = False,
        fpath_layout: Optional[StrOrPathLike] = None,
        logger: Optional[logging.Logger] = None,
        dry_run: bool = False,
    ):
        """Initialize the DICOM reorganization workflow."""
        super().__init__(
            dpath_root=dpath_root,
            name="dicom_reorg",
            fpath_layout=fpath_layout,
            logger=logger,
            dry_run=dry_run,
        )
        self.copy_files = copy_files

    def get_fpaths_to_reorg(
        self, participant: str, session: str, participant_first=True
    ) -> list[Path]:
        """
        Get file paths to reorganize for a single participant and session.

        This method can be overridden if the raw DICOM layout is different than what
        is typically expected.
        """
        # support both participant-first and session-first raw DICOM layouts
        if participant_first:
            dpath_downloaded = self.layout.dpath_raw_dicom / participant / session
        else:
            dpath_downloaded = self.layout.dpath_raw_dicom / session / participant

        # make sure directory exists
        if not dpath_downloaded.exists():
            raise FileNotFoundError(
                f"Raw DICOM directory not found for participant {participant}"
                f" session {session}: {dpath_downloaded}"
            )

        # crawl through directory tree and get all file paths
        fpaths = []
        for dpath, _, fnames in os.walk(dpath_downloaded):
            fpaths.extend(Path(dpath, fname) for fname in fnames)
        return fpaths

    def apply_fname_mapping(
        self, fname_source: str, participant: str, session: str
    ) -> str:
        """
        Apply a mapping to the file name.

        This method does not change the file name by default, but it can be overridden
        if the file names need to be changed during reorganization (e.g. for easier
        BIDS conversion).
        """
        return fname_source

    def run_single(self, participant: str, session: str):
        """Reorganize downloaded DICOM files for a single participant and session."""
        # get paths to reorganize
        # TODO add config option for session-first or participant-first raw DICOM layout
        fpaths_to_reorg = self.get_fpaths_to_reorg(
            participant, session, participant_first=False
        )

        # do reorg
        dpath_reorganized: Path = self.layout.dpath_sourcedata / participant / session
        self.mkdir(dpath_reorganized)
        for fpath_source in fpaths_to_reorg:
            fpath_dest = dpath_reorganized / self.apply_fname_mapping(
                fpath_source.name, participant=participant, session=session
            )

            # do not overwrite existing files
            if fpath_dest.exists():
                raise FileExistsError(
                    f"Cannot move file {fpath_source} to {fpath_dest}"
                    " because it already exists"
                )

            # either create symlinks or copy original files
            if not self.dry_run:
                if self.copy_files:
                    self.copy(fpath_source, fpath_dest)
                else:
                    fpath_source = os.path.relpath(fpath_source, fpath_dest.parent)
                    self.create_symlink(path_source=fpath_source, path_dest=fpath_dest)

        # update doughnut entry
        self.doughnut.set_status(
            participant=participant,
            session=session,
            col=self.doughnut.col_organized,
            status=True,
        )

    def run_main(self):
        """Reorganize all downloaded DICOM files."""
        for (
            participant,
            session,
        ) in self.doughnut.get_downloaded_participants_sessions():
            try:
                self.run_single(participant, session)
            except Exception as exception:
                self.logger.error(
                    "Error reorganizing DICOM files"
                    f" for participant {participant} session {session}: {exception}"
                )
