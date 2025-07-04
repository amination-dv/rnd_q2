import os
import sys
import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ilipy import Session
from ilipy.database import DistanceCorrelation

from rnd.utils.data_utils import (
    set_ili_run,
    generate_images,
    extract_bookmarks,
    generate_new_locations,
)


if __name__ == "__main__":
    # Example usage
    run_number = 4  # Change this to the desired run number
    _, _, inspection_id, env, _, start, end = set_ili_run(run_number)

    num_images = 10

    session = Session(environment=env)
    session.set_active_inspection(inspection_id)
    dist_corr = DistanceCorrelation(session)

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
        length=0.5,
        num_tracks=20,
        tick_sampling_interval=10,
        normalize=True,
        output_dir=f"data/fhr{run_number}/pos",
    )

    # # Saving no girth weld images
    no_gw_locations = generate_new_locations(
        all_gw_locations, range_limit=(start, end), num_new=num_images
    )
    saved_neg_files = generate_images(
        session=session,
        bookmark_locations=no_gw_locations,
        dist_corr=dist_corr,
        length=0.5,
        num_tracks=20,
        tick_sampling_interval=10,
        normalize=False,
        output_dir=f"data/fhr{run_number}/neg",
    )
