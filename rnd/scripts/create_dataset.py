import os
import sys
import random
import json
import glob


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ilipy import Session
from ilipy.database import DistanceCorrelation

from rnd.utils.data_utils import (
    set_ili_run,
    generate_images,
    extract_bookmarks,
    generate_new_locations,
    extract_dent_anomalies,
)


if __name__ == "__main__":
    # Example usage
    run_number = 4  # Change this to the desired run number
    num_images = 500
    _, _, inspection_id, env, _, start, end = set_ili_run(run_number, env="research")
    session = Session(environment=env)  # Pass "research" for positive samples
    session.set_active_inspection(inspection_id)
    dist_corr = DistanceCorrelation(session)


    dent_locations, tracks_ind = extract_dent_anomalies(
        session=session,
        inspection_id=inspection_id,
        dist_corr=dist_corr,
    )

    output_json_path = f"data/fhr{run_number}/dent_tracks_indices.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as json_file:
        json.dump(tracks_ind, json_file, indent=2)

    # qced_data_dir = "/home/zmirikha/GitHub/rnd_q2/data/fhr4/pos_0.5/qc"
    # dent_locations = []
    # for fn in glob.glob(os.path.join(qced_data_dir, "*.png")):
    #     dent_loc_mm = int(fn.split("/")[-1].split(".")[0])
    #     dent_locations.append(dent_loc_mm*1e-3)  # Convert mm to m

    # saved_dent_files = generate_images(
    #     session=session,
    #     bookmark_locations=dent_locations,
    #     dist_corr=dist_corr,
    #     length=0.3,
    #     num_tracks=20,
    #     tick_sampling_interval=10,
    #     normalize=True,
    #     output_dir=f"data/fhr{run_number}/pos_0.3_normalized",
    # )

    # Saving girth weld images
    all_gw_locations = extract_bookmarks(
        session=session,
        inspection_id=inspection_id,
        dist_corr=dist_corr,
        bookmark_type=0,
        anchored_only=False,  # Anchored girth welds
    )
    anchored_gw_locations = extract_bookmarks(
        session=session,
        inspection_id=inspection_id,
        dist_corr=dist_corr,
        bookmark_type=0,
        anchored_only=True,  # Anchored girth welds
    )

    sample_gw_locations = random.sample(
        anchored_gw_locations, min(num_images, len(anchored_gw_locations))
    )

    saved_gw_files = generate_images(
        session=session,
        bookmark_locations=sample_gw_locations,
        dist_corr=dist_corr,
        length=0.3,
        num_tracks=20,
        tick_sampling_interval=10,
        normalize=True,
        output_dir=f"data/fhr{run_number}/neg_0.3_normalized",
    )




    # # Saving no girth weld images
    no_gw_locations = generate_new_locations(
        all_gw_locations + dent_locations, range_limit=(start, end), num_new=num_images
    )
    saved_neg_files = generate_images(
        session=session,
        bookmark_locations=no_gw_locations,
        dist_corr=dist_corr,
        length=0.3,
        num_tracks=20,
        tick_sampling_interval=10,
        normalize=True,
        output_dir=f"data/fhr{run_number}/neg_0.3_normalized",
    )
    