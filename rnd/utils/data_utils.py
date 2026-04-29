# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ilipy import ClipTypes, OdometerTicks, OdometerTickRange, ViewDistance
from ilipy.features import Bookmarks
from ilipy.sensors import ArmAngleLookup
from ilipyutils.ml_features.query import FeatureQuery
from ilipyutils.ml_features.base import get_anomaly_types, AnomalyStatus
from ilipy.channeldata import  ImageProfile

# Default circumferential track count for create_arm_array / generate_images when num_tracks=None.
# Override process-wide with set_num_tracks() (e.g. from train_v4 --num-tracks).
_default_num_tracks: int = 20


def set_num_tracks(n: int) -> None:
    """Set default num_tracks for helpers that use None → global default."""
    global _default_num_tracks
    if int(n) < 1:
        raise ValueError("num_tracks must be >= 1")
    _default_num_tracks = int(n)


def get_num_tracks() -> int:
    return _default_num_tracks


def set_ili_run(run_number, env="research"):
    """
    Set the ILIT run configurations depending on the run.
    """
    run_configs = {
        1: {
            "surf_s3_bucket": "nan",
            "surf_s3_base_prefix": "nan",
            "inspection_id": "08VGYQQ1ZSU",
            "env": env,
            "clip_id": "nan",
            "start_distance": 100,
            "end_distance": 36400,
        },
        2: {
            "surf_s3_bucket": "nan",
            "surf_s3_base_prefix": "nan",
            "inspection_id": "09E27PISVFM",
            "env": env,
            "clip_id": "nan",
            "start_distance": 100,
            "end_distance": 36800,
        },
        3: {
            "surf_s3_bucket": "dv-fhr-3",
            "surf_s3_base_prefix": "track_runs/09FJN6N5AN6/ili_ml_surface/v1.2",
            "inspection_id": "09FJN6N5AN6",
            "env": env,
            "clip_id": "01-006-0TWUX9KZ",
            "start_distance": 100,
            "end_distance": 25700,
        },
        4: {
            "surf_s3_bucket": "dv-ilit0004",
            "surf_s3_base_prefix": "track_runs/09JBC62FLJZ/ili_ml_surface/v1.2",
            "inspection_id": "09JBC62FLJZ",
            "env": env,
            "clip_id": "01-017-0V9ZC3GT",
            "start_distance": 112796.8,
            "end_distance": 187770,
        },
        5: {
            "surf_s3_bucket": "dv-ilit0005",
            "surf_s3_base_prefix": "track_runs/09QWB8A52AN/ili_ml_surface/v1.2",
            "inspection_id": "09QWB8A52AN",
            "env": env,
            "clip_id": "01-017-0V9ZC3GT",
            "start_distance": 10,
            "end_distance": 13983,
        },
        6: {
            "surf_s3_bucket": "dv-ilit0006",
            "surf_s3_base_prefix": "track_runs/09WMV85VAMM/ili_ml_surface/v1.2",
            "inspection_id": "09WMV85VAMM",
            "env": env,
            "clip_id": "",
            "start_distance": -1,
            "end_distance": 1145,
        }
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
            continue  # clip can't map those distances

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
    bookmarks = Bookmarks(session=session)
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
    num_tracks: int | None = None,
    num_tick_samples: int = 500,
    normalize: str = "patch",
) -> np.ndarray:
    if num_tracks is None:
        num_tracks = get_num_tracks()
    all_dfs = pd.concat(arm_data_dict.values(), ignore_index=True)
    grouped = all_dfs.groupby("track_index")["arm_axis_angles_rad"]
    groups = {track_idx: group.values for track_idx, group in grouped}

    arm_array = np.zeros((num_tracks, num_tick_samples), dtype=np.float32)
    filled_rows = []
    for i in range(num_tracks):
        if i in groups:
            values = groups[i]
            length = min(num_tick_samples, len(values))
            arm_array[i, :length] = values[:length]
            if length < num_tick_samples and length > 0:
                arm_array[i, length:] = values[length - 1]
            filled_rows.append(i)
    rows = np.array(filled_rows, dtype=int)
    if normalize == "std":
        
        arm_array[rows] -= arm_array[rows].mean(axis=1, keepdims=True)
        stds = arm_array[rows].std(axis=1)
        ref_row = rows[np.argmax(stds)]
        ref_min = arm_array[ref_row].min()
        ref_max = arm_array[ref_row].max()
        denom = (ref_max - ref_min) if ref_max != ref_min else 1.0
        arm_array[rows] = (arm_array[rows] - ref_min) / denom
        
    elif normalize == "patch":

        row_min = arm_array[rows].min(axis=1, keepdims=True)
        row_max = arm_array[rows].max(axis=1, keepdims=True)
        denom = np.where((row_max - row_min) == 0, 1, row_max - row_min)
        arm_array[rows] = (arm_array[rows] - row_min) / denom
    
    

    return arm_array
        


def generate_images(
    session,
    bookmark_locations,
    dist_corr,
    stats,
    length: float = 0.5,
    num_tracks: int | None = None,
    tick_sampling_interval=10,
    normalize: str = "patch",
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
        normalize (str): Normalization method to apply ("patch", "range", or None).

    Returns:
        List[str]: File paths of saved matrices.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory '{output_dir}': {e}")

    if not bookmark_locations:
        raise ValueError("No valid bookmarks found.")

    if num_tracks is None:
        num_tracks = get_num_tracks()

    saved_files = []

    for view_dist, ind in tqdm(zip(bookmark_locations, stats)):
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
            matrix = create_arm_array(
                arm_data,
                num_tracks=num_tracks,
                num_tick_samples=num_tick_samples,
                normalize=normalize,
            )

            # Save with view distance in filename (rounded to 2 decimals)
            view_dist_mm = round(view_dist * 1000, 2)
            filename = f"d{view_dist_mm:010.0f}_{ind}.npy"
            #filepath = os.path.join(output_dir, status, filename)
            #qc_output_dir = os.path.join(output_dir, status,"qc")
            filepath = os.path.join(output_dir, filename)
            qc_output_dir = os.path.join(output_dir,"qc")
            os.makedirs(qc_output_dir, exist_ok=True)
            np.save(filepath, matrix)
            saved_files.append(filepath)

            plt.imshow(
                matrix.T, cmap="inferno", origin="lower", aspect="auto",
                interpolation="nearest",vmin=0, vmax=1
            )
            plt.colorbar()
            plt.axis("off")
            plt.savefig(
                os.path.join(qc_output_dir, f"{view_dist_mm:010.0f}_{ind}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        except Exception as e:
            print(f"Skipping view distance {view_dist:.2f} due to error: {e}")
            continue

    print(f"Saved {len(saved_files)} arm angle matrices to '{output_dir}'")
    return saved_files

def extract_dent_anomalies(session, inspection_id, dist_corr, status="Approved"):
    """
    Extract dent anomalies from the inspection session.

    Args:
        session (Session): The ILIPY session object.
        inspection_id (str): The ID of the inspection session.
        dist_corr (DistanceCorrelation): The distance correlation object.

    Returns:
        list: A list of view distances where dents are located.
    """
    bookmarks = Bookmarks(session.database_connector)
    feature_query = FeatureQuery(session=session, bookmarks_interface=bookmarks)
    session.set_active_inspection(inspection_id)
    locations = []
    track_inds = []
    def add_location(loc_list, val, tol):
        for existing in loc_list:
            if abs(existing - val) < tol:
                return False
        loc_list.append(val)
        return True

    # Get Dent Anomaly Type
    dent_anomaly_type = [a for a in get_anomaly_types() if "Dent" in a.name]
    clips = session.get_clips_by_type(ClipTypes.ChannelData)
    for dent_type in dent_anomaly_type:
        # Query clips with dent anomalies for the specified inspection
        clip_dent_dict = {}
        for clip in clips:
            if clip.odometer_tick_range.max.value - clip.odometer_tick_range.min.value < 1000:
                continue
            clip_dent_list_all=feature_query.get_anomalies_by_clip_by_anomaly_type(
                    clip_id=clip.clip_id,
                    inspection_id=inspection_id,
                    anomaly_type=dent_type,
                )
            clip_dent_list = [dent for dent in clip_dent_list_all if dent.status.value == status]
            if len(clip_dent_list) > 0:
                if clip.clip_id not in clip_dent_dict:
                    clip_dent_dict[clip.clip_id] = clip_dent_list
                else:
                    clip_dent_dict[clip.clip_id].extend(clip_dent_list)


        # Iterate through clips and dents
        for clip_id, dent_list in clip_dent_dict.items():
            for dent in dent_list:
                    #if dent.status.value == "Size Anomaly":
                        for track_loc in dent.feature_location.location_matrix:
                            for clip_loc in track_loc:
                                if clip_loc.clip.clip_id == clip_id:
                                    dent_odo_start, dent_odo_end = clip_loc.odometer_ticks_range
                                    dent_odo = (dent_odo_start+dent_odo_end)/2
                                    view_distance = dist_corr.get_view_distance_from_odometer_ticks(clip_loc.clip, OdometerTicks(int(dent_odo)))
                                    vd_val = view_distance.value
                                    min_sep=0.2
                                    add_location(locations, vd_val, min_sep)
                                    track_inds.append(clip_id.split("-")[1])

    return locations, track_inds

def extract_dent_anomalies_optimized(session, inspection_id, dist_corr, statuses=["Approved"]):
    """
    Extract dent anomalies from the inspection session.

    Args:
        session (Session): The ILIPY session object.
        inspection_id (str): The ID of the inspection session.
        dist_corr (DistanceCorrelation): The distance correlation object.
        statuses (list[str | int]): Status names or IDs to filter by
            (e.g. ["Approved", "Not Sized - Approved"]).

    Returns:
        tuple: (locations, track_inds, scan_angle_range, radial_position_mm_range,
                frame_indices_range, odometer_ticks_range)
    """
    from ilipy.features import AnomalyQuery

    bookmarks = Bookmarks(session)
    session.set_active_inspection(inspection_id)

    # Resolve status names to IDs
    all_status_types = bookmarks.get_anomaly_status_types()
    name_to_id = {st.name: st.anomaly_status_type_id for st in all_status_types}
    status_ids = []
    for s in statuses:
        if isinstance(s, int):
            status_ids.append(s)
        elif s in name_to_id:
            status_ids.append(name_to_id[s])
        else:
            raise ValueError(
                f"Unknown status '{s}'. Available: {list(name_to_id.keys())}"
            )

    # Fetch filtered anomalies via AnomalyQuery (much faster than get_anomalies)
    query = AnomalyQuery()
    query.statuses = status_ids
    query.tag_names = ["Dent-Detection-v3"]

    all_anomalies = []
    page_offset, page_size = 0, 300
    while True:
        page = bookmarks.filter_anomalies(inspection_id, query, page_offset, page_size)
        if not page:
            break
        all_anomalies.extend(page)
        if len(page) < page_size:
            break
        page_offset += page_size
    print(f"Fetched {len(all_anomalies)} anomalies matching status + tag filter")
    all_anomalies = [a for a in all_anomalies if "Dent-Detection-v3" in a.tags]
    target_type_names = {a.name for a in get_anomaly_types() if "Nominal" in a.name}
    anomaly_types_by_id = {a.anomaly_type_id: a.name for a in bookmarks.get_anomaly_types()}
    # Map identification_type_id -> anomaly type name (via anomaly_type_id)
    # ident_types = bookmarks.get_anomaly_identification_types()
    # ident_id_to_type_name = {
    #     it.anomaly_identification_type_id: anomaly_types_by_id.get(it.anomaly_type_id, "")
    #     for it in ident_types
    # }

    # Build set of valid clip IDs (clips with enough data)
    clips = session.get_clips_by_type(ClipTypes.ChannelData)
    valid_clip_ids = {
        clip.clip_id
        for clip in clips
        if clip.odometer_tick_range.max.value - clip.odometer_tick_range.min.value >= 1000
    }

    locations = []
    track_inds = []
    scan_angle_range = []
    radial_position_mm_range = []
    frame_indices_range = []
    odometer_ticks_range = []

    # Wrap each anomaly once
    from ilipyutils.ml_features.base import ilipy_info_to_wrap_info

    for anomaly in tqdm(all_anomalies, desc="Processing anomalies"):

        if anomaly.feature.anomaly_identification.name not in target_type_names:
            continue

        try:
            wrapped = ilipy_info_to_wrap_info(anomaly, session=session, bookmarks_interface=bookmarks)
        except Exception:
            continue

        if wrapped.feature_location is None:
            continue

        for track_loc in wrapped.feature_location.location_matrix:
            for clip_loc in track_loc:
                cid = clip_loc.clip.clip_id
                if cid not in valid_clip_ids:
                    continue
                dent_odo_start, dent_odo_end = clip_loc.odometer_ticks_range
                dent_odo = (dent_odo_start + dent_odo_end) / 2
                view_distance = dist_corr.get_view_distance_from_odometer_ticks(
                    clip_loc.clip, OdometerTicks(int(dent_odo))
                )
                locations.append(view_distance.value)
                track_inds.append(cid.split("-")[1])
                scan_angle_range.append(clip_loc.scan_angle_range)
                radial_position_mm_range.append(clip_loc.radial_position_mm_range)
                frame_indices_range.append(clip_loc.frame_indices_range_dict[ImageProfile.ZeroAngle])
                odometer_ticks_range.append(clip_loc.odometer_ticks_range)

    return locations, track_inds, scan_angle_range, radial_position_mm_range, frame_indices_range, odometer_ticks_range
