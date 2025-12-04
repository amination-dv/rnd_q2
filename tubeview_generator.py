from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from ilipy import LogLevel, PipeDistance, PipeDistanceRange, Session
from ilipyutils.tubeviews import MultiTrackTubeview, dataarray_to_image
from tqdm.auto import tqdm


def process_tubeviews(
    run_csv: str,
    output_dir: str,
    offset: float = 1.0,
    environment: str = "prod",
):
    """
    Process multi-track tubeviews from a CSV file.

    Args:
        run_csv: Path to the CSV file containing inspection data
        output_dir: Directory to save output images
        offset: Offset distance for tubeview extraction (default: 1.0)
        environment: Session environment (default: "prod")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load CSV
    run_df = pd.read_csv(run_csv)

    # Track component_id counts for indexing
    component_counts = {}

    # Initialize session
    session = Session(environment)
    session.settings.set_log_level(LogLevel.off)
    tracks = None  # all tracks
    print(
        "Found inspections to process:",
        (ninspections := run_df["inspection_id"].nunique()),
    )
    pbar = tqdm(run_df.groupby("inspection_id"), total=ninspections)

    for inspection_id, group in pbar:
        pbar.set_description(f"Inspection: {inspection_id}")
        session.set_active_inspection(inspection_id)
        nrows = len(group)
        if session.active_inspection.inspection_id != inspection_id:
            print(f"  Warning: Failed to set active inspection to {inspection_id}")
            continue
        # Create MultiTrackTubeview instance
        mttv = MultiTrackTubeview(
            session=session,
            tracks=tracks,
            tubeview_engine="TubeView",
            tubeview_engine_kwargs={"masking": "OD"},
        )
        for _, row in tqdm(group.iterrows(), total=nrows, leave=False):
            component_id = row["name"].lower()
            pipe_distance_center = row["pipeline_distance"]

            # Get or initialize count for this component
            if component_id not in component_counts:
                component_counts[component_id] = 0
            else:
                component_counts[component_id] += 1

            current_index = component_counts[component_id]

            # Create component-specific output directory
            component_dir = output_path / component_id
            component_dir.mkdir(parents=True, exist_ok=True)
            output_filename_nc = (
                component_dir / f"{inspection_id}_{component_id}_{current_index:03d}.nc"
            )
            output_filename_png = (
                component_dir
                / f"{inspection_id}_{component_id}_{current_index:03d}.png"
            )
            if output_filename_nc.exists():
                print(f"Skipping existing file: {output_filename_nc}")
                continue

            # print(
            #     f"Processing {component_id} (index: {current_index}) - Inspection: {inspection_id}"
            # )
            # Define distance range
            PD_START_M = pipe_distance_center - offset
            PD_END_M = pipe_distance_center + offset
            DISTANCE_RANGE = PipeDistanceRange(
                PipeDistance(PD_START_M), PipeDistance(PD_END_M)
            )
            PD_STEP_LENGTH = None  # PipeDistance(1.0)
            PD_STEP_PADDING = None
            resolution = PipeDistance(1.5e-3)

            # Process tubeviews
            tubeviews_da = mttv.process(
                distance_range=DISTANCE_RANGE,
                step_length=PD_STEP_LENGTH,
                step_padding=PD_STEP_PADDING,
                show_progress=False,
                max_workers=1,
            )

            # Rasterize tubeviews
            tubeviews_da = mttv.equalize_lateral(tubeviews_da)
            # tubeviews_da = mttv.rasterize(
            #     tubeviews_da,
            #     distance_range=DISTANCE_RANGE,
            #     axial_grid_step=resolution,
            #     max_gap_size=None,  # Ignore gaps
            #     agg="mean",
            # )
            tubeviews_da = mttv.interpolate(
                tubeviews_da,
                distance_range=DISTANCE_RANGE,
                axial_grid_step=resolution,
                max_gap_size="auto",
            )

            # Improving per-track amplitude balancing
            # ref_region = slice(
            #     round(len(next(iter(tubeviews_reg_da.values()))) * 0.9),
            #     None,
            # )
            # tubeviews_reg_da = {
            #     track: tubeview.fillna(0) / np.nanmedian(tubeview[ref_region, :])
            #     for track, tubeview in tubeviews_reg_da.items()
            # }
            # tubeviews_reg_hist_eq_da = mttv.equalize_histogram(
            #     tubeviews_reg_da,
            #     references={
            #         track: (da := tubeview[ref_region, :].to_numpy()[:])[np.isfinite(da)]
            #         for track, tubeview in tubeviews_reg_da.items()
            #     },
            # )

            # Combine and save full tubeview images
            if tubeviews_da == {}:
                print(f"  Warning: No tubeview data for {inspection_id}, skipping.")
                continue

            full_tubeview = mttv.combine(
                tubeviews_da,
                mapping="interpolate",
                y_coord="tool_lateral_deg",
                rasterize_agg="linear",
            )
            # full_tubeview.to_netcdf(output_filename_nc)
            full_tubeview_reg_hist_eq_PIL = dataarray_to_image(
                full_tubeview.sortby("tool_lateral_deg").T, how="eq_hist"
            ).to_pil(origin="upper")

            # Save with indexed filename
            full_tubeview_reg_hist_eq_PIL.save(output_filename_png)
            print(f"  Saved: {output_filename_nc}")

    print(f"\nProcessing complete! Processed {len(run_df)} tubeviews.")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process multi-track tubeviews from inspection data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv",
        type=str,
        help="Path to CSV file containing inspection data (must have columns: inspection_id, name, pipeline_distance)",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for saving tubeview images",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="Offset distance for tubeview extraction (in meters)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="prod",
        choices=["prod", "dev", "test", "research"],
        help="Session environment",
    )

    args = parser.parse_args()

    # Validate CSV file exists
    if not os.path.exists(args.csv):
        parser.error(f"CSV file not found: {args.csv}")

    # Run processing
    process_tubeviews(
        run_csv=args.csv,
        output_dir=args.output_dir,
        offset=args.offset,
        environment=args.env,
    )


if __name__ == "__main__":
    main()
