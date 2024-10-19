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


def get_mris_preproc_cmd(FS_dir, participant_id, out_file_suffix, meas="thickness", fwhm=0, template="fsaverage"):
    """ A function to generate FreeSurfer's mris_preproc command
    """
    
    fpath_lh = Path(f"{FS_dir}/{participant_id}/surf/lh.thickness")
    fpath_rh = Path(f"{FS_dir}/{participant_id}/surf/rh.thickness")
    fpath_status = Path.is_file(fpath_lh) & Path.is_file(fpath_rh)
    FS_CMD_dict = {}

    if fpath_status:
        participants_str = f"--s {participant_id}"
        
        for hemi in ["lh", "rh"]:
            d = os.path.dirname(out_file_suffix)
            f = os.path.basename(out_file_suffix)
            hemi_out_file = f"{d}/{participant_id}_{hemi}_{f}"

            FS_CMD = f"mris_preproc {participants_str} --target {template} --hemi {hemi} --meas {meas} --fwhm {fwhm} --out {hemi_out_file}"
            FS_CMD_dict[hemi] = FS_CMD
    
    else:
        print(f"ignoring {participant_id} with missing surf files...")
    
    return FS_CMD_dict
        

def run(FS_dir, participants_list, out_file_suffix, meas, fwhm, template):
    """ function to execute FS container with mris_preproc command
    """
    for participant_id in participants_list:
        FS_CMD_dict = get_mris_preproc_cmd(FS_dir, participant_id, out_file_suffix, meas, fwhm, template)

        if FS_CMD_dict:
            for hemi, FS_CMD in FS_CMD_dict.items():
                print(f"hemisphere: {hemi}")
                CMD_ARGS = SINGULARITY_CMD + FS_CMD 

                CMD = CMD_ARGS.split()

                print("-"*25)
                print(CMD_ARGS)
                print("-"*25)

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
    parser.add_argument('--output_dir', type=str, default=None, help='out_file path for the processed / aggregated output')
    parser.add_argument('--meas', type=str, default="thickness", help='cortical measure')
    parser.add_argument('--fwhm', type=int, default=10, help='smoothing kernel in mm')
    parser.add_argument('--template', type=str, default="fsaverage", help='freesurfer template (fsaverage or fsaverage5)')

    # TODO add overwrite flag

    args = parser.parse_args()

    global_config_file = args.global_config
    session_id = args.session_id
    
    output_dir = args.output_dir
    
    meas = args.meas
    fwhm = args.fwhm
    template = args.template

    session = f"ses-{session_id}"

    # Read global config
    with open(global_config_file, 'r') as f:
        global_configs = json.load(f)

    dataset_root = global_configs["DATASET_ROOT"]
    CONTAINER_STORE = global_configs["CONTAINER_STORE"]
    FS_VERSION = global_configs["PROC_PIPELINES"]["freesurfer"]["VERSION"]
    FS_CONTAINER = global_configs["PROC_PIPELINES"]["freesurfer"]["CONTAINER"]    
    FS_CONTAINER = FS_CONTAINER.format(FS_VERSION)
    FS_CONTAINER_PATH = f"{CONTAINER_STORE}{FS_CONTAINER}"

    # Paths
    FS_dir = f"{dataset_root}/derivatives/freesurfer/v{FS_VERSION}/output/{session}/" 
    FS_license = f"{FS_dir}/license.txt"

    if output_dir is None:
        output_dir = f"{dataset_root}/derivatives/freesurfer/v{FS_VERSION}/surfmaps/{session}/" 

    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)

    # grab bids_ids 
    manifest = f"{dataset_root}/manifest.csv"
    doughnut = f"{dataset_root}/scratch/raw_dicom/doughnut.csv"
        
    # Singularity CMD 
    SINGULARITY_CMD=f"singularity exec -B {FS_dir}:/fsdir -B {output_dir}:/output_dir {FS_CONTAINER_PATH} "

    # Read participant lists and filter by session and group
    doughnut_df = pd.read_csv(doughnut)
    doughnut_df = doughnut_df[doughnut_df["session"] == session]
    n_bids = len(doughnut_df["bids_id"].unique())

    print("")
    print("-"*50)
    print("Starting FS analysis...")
    print("-"*50)
    
    print(f"using session: {session}")
    print(f"number of available BIDS participants: {n_bids}")

    proc_participants = doughnut_df["bids_id"].unique() #.isin(group_participants)]["bids_id"].values
    n_proc_participants = len(proc_participants)
    print(f"number of proc particiants (bids with group info): {n_proc_participants}")

    if n_proc_participants > 0:
        print("")
        print("-"*50)
        print("Starting FS utils")

        out_file = f"/output_dir/surf_{meas}_{fwhm}mm.mgh"

        print("Running mris_preproc separately for each participant and for left and right hemisphere\n")
        run(FS_dir, proc_participants, out_file, meas, fwhm, template)
        
        print(" -"*30)
        print("")
        
        print("mris_preproc run complete")
        print("-"*50)
    else:
        print("-"*50)
        print("No participants found to process...")
        print("-"*50)