# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from typing import Dict

from ilipy import ClipTypes, Session, OdometerTicks, OdometerTickRange, ViewDistance
from ilipy.database import DistanceCorrelation
from ilipy.features import Bookmarks
from ilipy.sensors import ArmAngleLookup


def set_ili_run(run_number):
    """
    Set the ILIT run configurations depending on the run.
    """
    run_configs = {
        1: {
            "surf_s3_bucket": "nan",
            "surf_s3_base_prefix": "nan",
            "inspection_id": "08VGYQQ1ZSU",
            "env": "research",
            "clip_id": "nan",
            "start_distance": 100,
            "end_distance": 36400,
        },
        2: {
            "surf_s3_bucket": "nan",
            "surf_s3_base_prefix": "nan",
            "inspection_id": "09E27PISVFM",
            "env": "research",
            "clip_id": "nan",
            "start_distance": 100,
            "end_distance": 36800,
        },
        3: {
            "surf_s3_bucket": "dv-fhr-3",
            "surf_s3_base_prefix": "track_runs/09FJN6N5AN6/ili_ml_surface/v1.2",
            "inspection_id": "09FJN6N5AN6",
            "env": "research",
            "clip_id": "01-006-0TWUX9KZ",
            "start_distance": 100,
            "end_distance": 25700,
        },
        4: {
            "surf_s3_bucket": "dv-ilit0004",
            "surf_s3_base_prefix": "track_runs/09JBC62FLJZ/ili_ml_surface/v1.2",
            "inspection_id": "09JBC62FLJZ",
            "env": "prod",
            "clip_id": "01-017-0V9ZC3GT",
            "start_distance": 112796.8,
            "end_distance": 187770,
        },
        5: {
            "surf_s3_bucket": "dv-ilit0005",
            "surf_s3_base_prefix": "track_runs/09QWB8A52AN/ili_ml_surface/v1.2",
            "inspection_id": "09QWB8A52AN",
            "env": "prod",
            "clip_id": "01-017-0V9ZC3GT",
            "start_distance": 10,
            "end_distance": 13983,
        },
    }

    if run_number not in run_configs:
        raise ValueError(f"Invalid run_number: {run_number}")

    config = run_configs[run_number]
    return (
        config["surf_s3_bucket"],
        config["surf_s3_base_prefix"],
        config["inspection_id"],
        config["env"],
        config["clip_id"],
        config["start_distance"],
        config["end_distance"],
    )


def extract_arm_angles(
    session, dist_corr, view_distance_range_m, tick_sampling_interval=10
):
    """
    Processes inspection data by sampling odometer ticks within a specified view distance range
    and extracting relevant metrics for each clip.
    Args:
        env (Environment): The environment object representing the current simulation or system state.
        inspection_id (str): The unique identifier for the inspection session.
        view_distance_range_m (tuple): A tuple (start_vd, end_vd) specifying the range of view distances
            in meters to process.
        tick_sampling_interval (int, optional): The interval (in ticks) at which to sample odometer ticks
            within the overlapping range. Defaults to 10.
    Returns:
        dict: A dictionary where each key is a clip ID, and the value is a pandas DataFrame containing
            sampled data. Each DataFrame includes the following columns:
            - "track_index": The index of the track associated with the clip.
            - "odometer_tick": The sampled odometer tick.
            - "view_distances_m": The view distance in meters at the sampled tick.
            - "arm_axis_angles_rad": The arm axis angle in radians at the sampled tick.
    Notes:
        - Clips are filtered based on their overlap with the specified view distance range.
        - Sampling is performed only within the overlapping tick range for each clip.
        - Clips that cannot map the specified view distances or do not yield any samples are skipped.

    """
    all_arms = {}
    clips = session.get_clips_by_type(ClipTypes.ChannelData)

    start_vd, end_vd = view_distance_range_m

    for clip in clips:
        # 1) map view-distances -> odometer-ticks for this clip
        try:
            tick_start = dist_corr.get_odometer_ticks_from_view_distance(
                clip, ViewDistance(start_vd)
            )
            tick_end = dist_corr.get_odometer_ticks_from_view_distance(
                clip, ViewDistance(end_vd)
            )
        except Exception:
            continue  # clip can’t map those distances

        query_range = OdometerTickRange(tick_start, tick_end)
        clip_range = clip.odometer_tick_range

        # 2) skip clips with no raw-tick overlap
        if (
            clip_range.max.value < query_range.min.value
            or clip_range.min.value > query_range.max.value
        ):
            continue

        # 3) clamp to the actual overlapping integer ticks
        lo = max(clip_range.min.value, tick_start.value)
        hi = min(clip_range.max.value, tick_end.value)
        if hi <= lo:
            continue

        # 4) sample every tick_sampling_interval ticks
        arm_lookup = ArmAngleLookup(session, clip)
        rows = []
        for tick in range(int(lo), int(hi) + 1, tick_sampling_interval):
            odot = OdometerTicks(tick)
            angle = arm_lookup.get_axis_angle_from_odometer_ticks(odot)
            # vd_m = dist_corr.get_view_distance_from_odometer_ticks(clip, odot).value
            rows.append(
                {
                    "track_index": clip.track_index,
                    # "odometer_tick": tick,
                    # "view_distances_m": vd_m,
                    "arm_axis_angles_rad": angle,
                }
            )

        # 5) only keep clips where we actually got samples
        if rows:
            all_arms[clip.clip_id] = pd.DataFrame(rows)

    return all_arms


# pick which bookmark_type you want to sample:
#   0 = girth welds  (will only keep anchored ones)
#   1 = bends
#   2 = Tees
def extract_bookmarks(
    session, inspection_id, dist_corr, bookmark_type=0, anchored_only=True
):
    bookmarks = Bookmarks(session.database_connector)
    components = bookmarks.get_components(inspection_id)

    locations = []

    for component in components:
        feature_type_id = component.feature.component_type_id

        if feature_type_id != bookmark_type:
            continue

        pipe_dist = component.feature.pipeline_distance
        view_dist = dist_corr.get_view_distance_from_pipe_distance(pipe_dist)

        if view_dist is None:
            continue

        anchored = (
            dist_corr.is_anchored(component.feature) if feature_type_id == 0 else False
        )

        if anchored_only and not anchored:
            continue
        # TODO: validate that view_dist is within the range of the run

        # locations.append((pipe_dist.value, view_dist.value))
        locations.append(view_dist.value)

    return locations


def generate_new_locations(
    existing_locations,
    num_new=100,
    range_limit=(0, 50000),
    min_distance=1,
    max_attempts=10000,
):
    """
    Generates `num_new` locations not within `min_distance` of any existing or new location.

    Parameters:
        existing_locations (list of float): Original spots (in meters).
        num_new (int): Number of new spots to generate.
        range_limit (tuple of int): Range (start, end) for location generation in meters.
        min_distance (float): Minimum spacing in meters.
        max_attempts (int): Safety limit for iterations.

    Returns:
        list of float: New valid locations.

    Raises:
        RuntimeError: If unable to find enough valid locations.
    """
    new_locations = []
    all_locations = existing_locations.copy()

    start, end = range_limit
    attempts = 0
    while len(new_locations) < num_new:
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to generate enough locations. Try lowering density or constraints."
            )

        candidate = round(random.uniform(start, end), ndigits=2)
        if all(abs(candidate - loc) >= min_distance for loc in all_locations):
            new_locations.append(candidate)
            all_locations.append(candidate)
        attempts += 1

    return new_locations


def create_arm_array(
    arm_data_dict: Dict[str, pd.DataFrame],
    num_tracks: int = 20,
    num_tick_samples: int = 500,
    normalize: bool = False,
) -> np.ndarray:
    all_dfs = pd.concat(arm_data_dict.values(), ignore_index=True)
    grouped = all_dfs.groupby("track_index")["arm_axis_angles_rad"]
    groups = {track_idx: group.values for track_idx, group in grouped}

    arm_array = np.zeros((num_tracks, num_tick_samples), dtype=np.float32)

    for i in range(num_tracks):
        if i in groups:
            values = groups[i]
            length = min(num_tick_samples, len(values))
            arm_array[i, :length] = values[:length]
            if length < num_tick_samples and length > 0:
                arm_array[i, length:] = values[length - 1]

    if normalize:
        row_min = arm_array.min(axis=1, keepdims=True)
        row_max = arm_array.max(axis=1, keepdims=True)
        denom = np.where((row_max - row_min) == 0, 1, row_max - row_min)
        arm_array = (arm_array - row_min) / denom

    return arm_array


def generate_images(
    session,
    bookmark_locations,
    dist_corr,
    length: float = 0.5,
    num_tracks: int = 20,
    tick_sampling_interval=10,
    normalize: bool = False,
    output_dir: str = "arm_angles",
):
    """
    Extracts and saves individual arm angle matrices from positive bookmark view distances.

    Args:
        env: The simulation environment.
        inspection_id (str): Inspection session ID.
        bookmark_type (int): Type of bookmark (0 = anchored girth welds).
        range_half_width (float): Half-width of the range around each view distance.
        num_tracks (int): Number of tracks per matrix.
        fixed_length (int): Fixed length of each arm angle row.
        output_dir (str): Directory where .npy files will be saved.

    Returns:
        List[str]: File paths of saved matrices.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory '{output_dir}': {e}")

    if not bookmark_locations:
        raise ValueError("No valid bookmarks found.")

    saved_files = []

    for view_dist in tqdm(bookmark_locations):
        # for view_dist in bookmark_locations:
        try:
            arm_data = extract_arm_angles(
                session=session,
                dist_corr=dist_corr,
                view_distance_range_m=(
                    view_dist - length / 2,
                    view_dist + length / 2,
                ),
                tick_sampling_interval=tick_sampling_interval,
            )
            if not arm_data:
                print(f"No arm data found at view distance {view_dist:.3f} m")
                continue

            num_tick_samples = int(length * 10000 / tick_sampling_interval)
            print(num_tick_samples)
            matrix = create_arm_array(
                arm_data,
                num_tracks=num_tracks,
                num_tick_samples=num_tick_samples,
                normalize=normalize,
            )

            # Save with view distance in filename (rounded to 2 decimals)
            view_dist_mm = round(view_dist * 1000, 2)
            filename = f"d{view_dist_mm:010.0f}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, matrix)
            saved_files.append(filepath)

        except Exception as e:
            print(f"Skipping view distance {view_dist:.2f} due to error: {e}")
            continue

    print(f"Saved {len(saved_files)} arm angle matrices to '{output_dir}'")
    return saved_files


# if __name__ == "__main__":
#     # Example usage
#     run_number = 4  # Change this to the desired run number
#     _, _, inspection_id, env, _, start, end = set_ili_run(run_number)

#     session = Session(environment=env)
#     session.set_active_inspection(inspection_id)
#     dist_corr = DistanceCorrelation(session)
#     bookmark_locations = extract_bookmarks(
#         session=session,
#         inspection_id=inspection_id,
#         dist_corr=dist_corr,
#         bookmark_type=0,  # Anchored girth welds
#     )

#     generate_images(
#         session=session,
#         bookmark_locations=bookmark_locations,
#         dist_corr=dist_corr,
#         length=0.5,
#         num_tracks=20,
#         fixed_length=500,
#         output_dir="tees",
#     )
