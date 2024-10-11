import os
import subprocess
import argparse
from pathlib import Path
import shutil
import pandas as pd
import json

# Make sure FS_LICENSE is defined in the container.
os.environ['SINGULARITYENV_FS_LICENSE'] = "/fsdir/license.txt"
os.environ['SINGULARITYENV_SUBJECTS_DIR'] = "/fsdir/"


def get_mris_preproc_cmd(FS_dir, participants_list, out_file, meas="thickness", fwhm=0, template="fsaverage"):
    """ A function to generate FreeSurfer's mris_preproc command
    """
    participants_str_list = []
    for participant in participants_list:
        fpath_lh = Path(f"{FS_dir}/{participant}/surf/lh.thickness")
        fpath_rh = Path(f"{FS_dir}/{participant}/surf/rh.thickness")
        fpath_status = Path.is_file(fpath_lh) & Path.is_file(fpath_rh)
        if fpath_status:
            participants_str_list.append(f"--s {participant}")
        else:
            print(f"ignoring {participant} with missing surf files...")

    participants_str = ' '.join(participants_str_list)
    FS_CMD_dict = {}
    for hemi in ["lh", "rh"]:
        d = os.path.dirname(out_file)
        f = os.path.basename(out_file)
        hemi_out_file = f"{d}/{hemi}_{f}"

        FS_CMD = f"mris_preproc {participants_str} --target {template} --hemi {hemi} --meas {meas} --fwhm {fwhm} --out {hemi_out_file}"
        FS_CMD_dict[hemi] = FS_CMD

    return FS_CMD_dict
        

def run(FS_dir, participants_list, out_file, meas, fwhm, template):
    """ function to execute FS container with mris_preproc command
    """
    FS_CMD_dict = get_mris_preproc_cmd(FS_dir, participants_list, out_file, meas, fwhm, template)
    for hemi, FS_CMD in FS_CMD_dict.items():
        print(f"hemisphere: {hemi}")
        CMD_ARGS = SINGULARITY_CMD + FS_CMD 

        CMD = CMD_ARGS.split()

        try:        
            proc = subprocess.run(CMD)

        except Exception as e:
            print(f"mris_preproc run failed with exceptions: {e}")

        print("-"*30)
        print("")

if __name__ == '__main__':
    # argparse
    HELPTEXT = """
    Script to perform DICOM to BIDS conversion using HeuDiConv
    """
    parser = argparse.ArgumentParser(description=HELPTEXT)
    parser.add_argument('--global_config', type=str, help='path to global config file for your nipoppy dataset', required=True)
    parser.add_argument('--session_id', type=str, help='session_id', required=True)
    parser.add_argument('--visit_id', type=str, default=None, help='visit_id')
    parser.add_argument('--group', type=str, default=None, help='filter participants based on a specific group value in the csv')
    parser.add_argument('--output_dir', type=str, default=None, help='out_file path for the processed / aggregated output')
    parser.add_argument('--meas', type=str, default="thickness", help='cortical measure')
    parser.add_argument('--fwhm', type=int, default=10, help='smoothing kernel in mm')
    parser.add_argument('--template', type=str, default="fsaverage", help='freesurfer template (fsaverage or fsaverage5)')

    args = parser.parse_args()

    global_config_file = args.global_config
    session_id = args.session_id
    visit_id = args.visit_id
    group = args.group
    output_dir = args.output_dir
    
    meas = args.meas
    fwhm = args.fwhm
    template = args.template

    session = f"ses-{session_id}"

    if visit_id is None:
        visit = f"V{session_id}"
    else:
        visit = f"V{visit_id}"

    # Read global config
    with open(global_config_file, 'r') as f:
        global_configs = json.load(f)

    dataset_root = global_configs["DATASET_ROOT"]
    CONTAINER_STORE = global_configs["CONTAINER_STORE"]
    FMRIPREP_CONTAINER = global_configs["PROC_PIPELINES"]["fmriprep"]["CONTAINER"]
    FMRIPREP_VERSION = global_configs["PROC_PIPELINES"]["fmriprep"]["VERSION"]
    FS_VERSION = global_configs["PROC_PIPELINES"]["freesurfer"]["VERSION"]
    FMRIPREP_CONTAINER = FMRIPREP_CONTAINER.format(FMRIPREP_VERSION)
    SINGULARITY_FMRIPREP = f"{CONTAINER_STORE}{FMRIPREP_CONTAINER}"

    # Paths
    FS_dir = f"{dataset_root}/derivatives/freesurfer/v{FS_VERSION}/output/{session}/" 
    FS_license = f"{FS_dir}/license.txt"

    

    if output_dir is None:
        output_dir = f"{dataset_root}/derivatives/freesurfer/v{FS_VERSION}/surfmaps/{session}/" 

    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)

    # grab bids_ids 
    manifest = f"{dataset_root}/manifest.csv"

    # grab Dx info
    demographics_csv = f"{dataset_root}/tabular/demographics.csv"
        
    # Singularity CMD 
    SINGULARITY_CMD=f"singularity exec -B {FS_dir}:/fsdir -B {output_dir}:/output_dir {SINGULARITY_FMRIPREP} "

    
    # --------------------------------------------------------------------
    # Replace this with doughnut
    # --------------------------------------------------------------------
    # Read participant lists and filter by session and group
    manifest_df = pd.read_csv(manifest)
    manifest_df = manifest_df[manifest_df["session"] == session]

    if "bids_id" not in manifest_df.columns:
        # Append bids id col from id_map_file (participant_id --> bids_id)
        id_mapping_csv = f"{dataset_root}/scratch/participant_id_bids_id_map.csv"
        id_mapping_df = pd.read_csv(id_mapping_csv)
        manifest_df = pd.merge(manifest_df, id_mapping_df[["participant_id","bids_id"]], on=["participant_id"])
    
    manifest_df = manifest_df[~manifest_df["bids_id"].isna()]
    n_bids = len(manifest_df["bids_id"].unique())
    # --------------------------------------------------------------------


    print("")
    print("-"*50)
    print("Starting FS analysis...")
    print("-"*50)
    
    print(f"using session: {session} and visit: {visit}")
    print(f"number of available BIDS participants: {n_bids}")

    demographics_df = pd.read_csv(demographics_csv)
    demographics_df = demographics_df[demographics_df["visit"] == visit]

    if group is None:
        print("No group filter specified, concatenating all participants")
        group = "all"
        group_participants = demographics_df["participant_id"].unique()
    else:
        print(f"Using {group} subset of participants")
        group_participants = demographics_df[demographics_df["group"] == group]["participant_id"].unique()

    n_group = len(group_participants)
    print(f"number of available group participants: {n_group}")

    proc_participants = manifest_df[manifest_df["participant_id"].isin(group_participants)]["bids_id"].values
    n_proc_participants = len(proc_participants)
    print(f"number of proc particiants (bids with group info): {n_proc_participants}")

    if n_proc_participants > 0:
        print("")
        print("-"*50)
        print("Starting FS utils")

        out_file = f"/output_dir/surf_concat_{group}_{fwhm}mm.mgh"

        print("Running mris_preproc separately for left and right hemisphere\n")
        run(FS_dir, proc_participants, out_file, meas, fwhm, template)
        
        print(" -"*30)
        print("")
        
        print("mris_preproc run complete")
        print("-"*50)
    else:
        print("-"*50)
        print("No participants found to process...")
        print("-"*50)