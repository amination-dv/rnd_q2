from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
from ilipy import LogLevel, PipeDistance, PipeDistanceRange, Session

from ilipyutils.tubeviews import MultiTrackTubeview, dataarray_to_image


def process_tubeviews(
    run_csv: str,
    output_dir: str,
    offset: float = 0.5,
    environment: str = "prod",
):
    """
    Process multi-track tubeviews from a CSV file.
    
    Args:
        run_csv: Path to the CSV file containing inspection data
        output_dir: Directory to save output images
        offset: Offset distance for tubeview extraction (default: 0.5)
        environment: Session environment (default: "prod")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    run_df = pd.read_csv(run_csv)
    
    # Track component_id counts for indexing
    component_counts = {}
    
    for _, row in run_df.iterrows():
        inspection_id = row["inspection_id"]
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
        
        print(f"Processing {component_id} (index: {current_index}) - Inspection: {inspection_id}")
        
        # Initialize session
        session = Session(environment)
        session.settings.set_log_level(LogLevel.off)
        session.set_active_inspection(inspection_id)
        tracks = None  # all tracks

        # Create MultiTrackTubeview instance
        mttv = MultiTrackTubeview(
            session=session,
            tracks=tracks,
            tubeview_engine="TubeView",
            tubeview_engine_kwargs={"masking": "OD"},
        )

        # Define distance range
        PD_START_M = pipe_distance_center - offset
        PD_END_M = pipe_distance_center + offset
        DISTANCE_RANGE = PipeDistanceRange(PipeDistance(PD_START_M), PipeDistance(PD_END_M))
        PD_STEP_LENGTH = None  # PipeDistance(1.0)
        PD_STEP_PADDING = None
        resolution = PD_STEP_LENGTH or PipeDistance(1.5e-3)

        # Process tubeviews
        tubeviews_da = mttv.process(
            distance_range=DISTANCE_RANGE,
            step_length=PD_STEP_LENGTH,
            step_padding=PD_STEP_PADDING,
            show_progress=True,
            max_workers=4,
        )

        # Rasterize tubeviews
        tubeview_cheq_da = mttv.equalize_lateral(tubeviews_da)
        tubeviews_reg_da = mttv.rasterize(
            tubeview_cheq_da,
            distance_range=DISTANCE_RANGE,
            step_length=resolution,
            max_gap_size=None,  # Ignore gaps
            agg="mean",
        )

        # Improving per-track amplitude balancing
        ref_region = slice(
            round(len(next(iter(tubeviews_reg_da.values()))) * 0.9),
            None,
        )
        tubeviews_reg_da = {
            track: tubeview.fillna(0) / np.nanmedian(tubeview[ref_region, :])
            for track, tubeview in tubeviews_reg_da.items()
        }
        tubeviews_reg_hist_eq_da = mttv.equalize_histogram(
            tubeviews_reg_da,
            references={
                track: (da := tubeview[ref_region, :].to_numpy()[:])[np.isfinite(da)]
                for track, tubeview in tubeviews_reg_da.items()
            },
        )

        # Combine and save full tubeview images
        full_tubeview_reg_hist_eq_da = mttv.combine(
            tubeviews_reg_hist_eq_da,
            mapping="interpolate",
            y_coord="tool_lateral_deg",
            rasterize_agg="cubic",
        )
        full_tubeview_reg_hist_eq_PIL = dataarray_to_image(
            full_tubeview_reg_hist_eq_da.sortby("tool_lateral_deg").T, how="eq_hist"
        ).to_pil(origin="upper")
        
        # Save with indexed filename
        output_filename = component_dir / f"tubeview_{current_index:03d}.png"
        full_tubeview_reg_hist_eq_PIL.save(output_filename)
        print(f"  Saved: {output_filename}")
    
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
        default=0.5,
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