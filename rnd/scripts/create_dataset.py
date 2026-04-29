import os
import sys
import random
import json
import glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ilipy import Session
from ilipy.database import DistanceCorrelation

from rnd.utils.data_utils import (
    set_ili_run,
    generate_images,
    extract_bookmarks,
    generate_new_locations,
    extract_dent_anomalies_optimized,
)
inspection_id = ["0AGBXB4WEGN"]#["0ABP0TFUSH1"]# ["0A1OLRV96N3", "0A49KLT3B7Y", "0ABP0TFUSH1"]
env = "prod"
#status = ["False Positive"]
status_list = ["Not Sized - Approved"]
tag_list = ["Dent-Detection-v3"]
num_images = 500
session = Session(environment=env)

def merge_close_locations(locations, track_indices, distance_threshold):
    """
    Combine locations that are closer than distance_threshold, calculate their mean,
    and group their corresponding track indices.
    
    Args:
        locations: list of location values (distances/positions)
        track_indices: list of track indices corresponding to locations
        distance_threshold: maximum distance between locations to merge them
    
    Returns:
        dict: {mean_location: [list of track_indices]} for each merged group
    """
    if len(locations) != len(track_indices):
        raise ValueError("locations and track_indices must have the same length")
    
    if len(locations) == 0:
        return {}
    
    # Convert to numpy arrays
    locations = np.array(locations)
    track_indices = np.array(track_indices)
    
    # Remove NaN values
    valid_mask = ~np.isnan(locations)
    locations = locations[valid_mask]
    track_indices = track_indices[valid_mask]
    
    if len(locations) == 0:
        return {}
    
    # Sort by location value
    sort_indices = np.argsort(locations)
    sorted_locations = locations[sort_indices]
    sorted_track_indices = track_indices[sort_indices]
    
    # Group consecutive locations that are within distance_threshold
    groups = []
    current_group_indices = [0]
    
    for i in range(1, len(sorted_locations)):
        # Check if current location is within threshold of the last location in current group
        last_idx_in_group = current_group_indices[-1]
        distance = sorted_locations[i] - sorted_locations[last_idx_in_group]
        
        if distance <= distance_threshold:
            # Within threshold, add to current group
            current_group_indices.append(i)
        else:
            # Beyond threshold, save current group and start new one
            groups.append(current_group_indices)
            current_group_indices = [i]
    
    # Add the last group
    groups.append(current_group_indices)
    
    # Create output dictionary: {mean_location: [track_indices]}
    result = {}
    
    for group_indices in groups:
        # Get locations and track indices for this group
        group_locations = sorted_locations[group_indices]
        group_track_indices = sorted_track_indices[group_indices].tolist()
        
        # Calculate mean location
        mean_location = float(np.mean(group_locations))
        
        # Store in result dictionary
        result[mean_location] = group_track_indices
    
    return result

if __name__ == "__main__":
    # Example usage

    for inspection in inspection_id:
        output_dir = f"data/{inspection}"
        session.set_active_inspection(inspection)
        dist_corr = DistanceCorrelation(session)
        

        dent_locations, tracks_ind, scan_angle_range, radial_position_mm_range, frame_indices_range, odometer_ticks_range = extract_dent_anomalies_optimized(
            session=session,
            inspection_id=inspection,
            dist_corr=dist_corr,
            statuses=status_list,
        )

        multi_track_dent_dict = merge_close_locations(dent_locations, tracks_ind, 0.2)
        multi_track_scan_angle_range_dict = merge_close_locations(dent_locations, scan_angle_range, 0.2)
        multi_track_radial_position_mm_range_dict = merge_close_locations(dent_locations, radial_position_mm_range, 0.2)
        multi_track_frame_indices_range_dict = merge_close_locations(dent_locations, frame_indices_range, 0.2)
        multi_track_odometer_ticks_range_dict = merge_close_locations(dent_locations, odometer_ticks_range, 0.2)

        #save dent locations and their tracks indices and other metadata after grouping themto json
  # Create structured data for each merged dent location
        dent_data_list = []
        
        # Get all unique merged locations (they should be the same across all dicts)
        merged_locations = set(multi_track_dent_dict.keys())
        merged_locations.update(multi_track_scan_angle_range_dict.keys())
        merged_locations.update(multi_track_radial_position_mm_range_dict.keys())
        merged_locations.update(multi_track_frame_indices_range_dict.keys())
        merged_locations.update(multi_track_odometer_ticks_range_dict.keys())
        
        # Iterate through merged dent locations
        for location in sorted(merged_locations):
            dent_entry = {
                "location": float(location),
                "track_indices": multi_track_dent_dict.get(location, []),
                "scan_angle_range": multi_track_scan_angle_range_dict.get(location, []),
                "radial_position_mm_range": multi_track_radial_position_mm_range_dict.get(location, []),
                "frame_indices_range": multi_track_frame_indices_range_dict.get(location, []),
                "odometer_ticks_range": multi_track_odometer_ticks_range_dict.get(location, [])
            }
            dent_data_list.append(dent_entry)
        
        # Save to JSON
        output_json_path = f"data/{inspection}/dent_locations_and_tracks_indices.json"
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as json_file:
            json.dump({
                "inspection_id": inspection,
                "dent_count": len(dent_data_list),
                "dents": dent_data_list
            }, json_file, indent=2)
        
        print(f"Saved {len(dent_data_list)} merged dent locations to {output_json_path}")


        saved_dent_files = generate_images(
            session=session,
            bookmark_locations=list(multi_track_dent_dict.keys()),
            dist_corr=dist_corr,
            stats=list(multi_track_dent_dict.values()),
            length=0.3, 
            num_tracks=22,
            tick_sampling_interval=10,
            normalize="std",
            output_dir=f"{output_dir}/false_positive/std_normalized",
        )
        saved_dent_files = generate_images(
            session=session,
            bookmark_locations=list(multi_track_dent_dict.keys()),
            dist_corr=dist_corr,
            stats= list(multi_track_dent_dict.values()),
            length=0.3,
            num_tracks=22,
            tick_sampling_interval=10,
            normalize="patch",
            output_dir=f"{output_dir}/false_positive/patch_normalized",
            )

    # Saving girth weld images
    # all_gw_locations = extract_bookmarks(
    #     session=session,
    #     inspection_id=inspection_id,
    #     dist_corr=dist_corr,
    #     bookmark_type=0,
    #     anchored_only=False,  # Anchored girth welds
    # )
    # anchored_gw_locations = extract_bookmarks(
    #     session=session,
    #     inspection_id=inspection_id,
    #     dist_corr=dist_corr,
    #     bookmark_type=0,
    #     anchored_only=True,  # Anchored girth welds
    # )

    # sample_gw_locations = random.sample(
    #     anchored_gw_locations, min(num_images, len(anchored_gw_locations))
    # )

    # saved_gw_files = generate_images(
    #     session=session,
    #     bookmark_locations=sample_gw_locations,
    #     dist_corr=dist_corr,
    #     length=0.3,
    #     num_tracks=20,
    #     tick_sampling_interval=10,
    #     normalize=True,
    #     output_dir=f"data/fhr{run_number}/neg_0.3_normalized",
    # )

    # # Saving no girth weld images
    # no_gw_locations = generate_new_locations(
    #     all_gw_locations + dent_locations, range_limit=(start, end), num_new=num_images
    # )


    # qced_data_dir = "/home/zmirikha/GitHub/rnd_q2/data/fhr4/final_data_binary_classifier/neg"
    # no_gw_locations = []
    # for fn in glob.glob(os.path.join(qced_data_dir, "*.npy")):
    #     loc_mm = int(fn.split("/")[-1].split(".")[0][1:])
    #     no_gw_locations.append(loc_mm*1e-3)  # Convert mm to m
    # saved_neg_files = generate_images(
    #     session=session,
    #     bookmark_locations=no_gw_locations,
    #     dist_corr=dist_corr,
    #     length=0.3,
    #     num_tracks=20,
    #     tick_sampling_interval=10,
    #     normalize="",
    #     output_dir=f"data/fhr{run_number}/neg_not_normalized",
    # )
    