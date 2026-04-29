"""Streamlit Web UI for Semi-Automatic QC Pipeline."""

import re
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from config import (
    PROBABILITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    NUM_LABELS,
    QC_RESULTS_DIR,
    QC_RESULTS_FILE,
    BASE_DATA_PATH,
)
from data_manager import DataManager, QCStatus, Sample
from inference import load_model, run_inference

st.set_page_config(
    page_title="QC Pipeline",
    page_icon="🔍",
    layout="wide",
)

# Session state key for selected data folder
DATA_FOLDER_KEY = "qc_data_folder"


def _normalize_data_folder_path(raw: str) -> Path:
    r"""Convert path to canonical form so same folder always uses same QC results.
    Handles Windows UNC (e.g. \\wsl.localhost\Ubuntu-22.04\home\...) -> /home/...
    """
    raw = raw.strip().replace("\\", "/")
    # Windows UNC for WSL: //wsl.localhost/Ubuntu-22.04/home/... or /wsl.localhost/.../home/...
    if "wsl.localhost" in raw.lower() and "/home/" in raw:
        # Extract /home/... and use that so key matches Linux path
        i = raw.lower().index("/home/")
        raw = raw[i:]
    path = Path(raw).expanduser().resolve()
    return path


def _sanitize_folder_name(path: Path) -> str:
    """Create a safe subfolder name from path (e.g. for qc_results)."""
    s = path.resolve().as_posix().strip("/")
    s = re.sub(r"[^\w\-.]", "_", s)
    return s[:80] if len(s) > 80 else s or "default"


def get_data_folder_path() -> Path | None:
    """Return currently selected data folder path, or None if not set."""
    raw = st.session_state.get(DATA_FOLDER_KEY)
    if not raw:
        return None
    return _normalize_data_folder_path(raw)


def paths_from_base(base: Path):
    """Build QC paths from base data folder (e.g. .../Approved)."""
    base = base.resolve()
    png_std_qc = base / "std_normalized" / "qc"
    png_patch_qc = base / "patch_normalized" / "qc"
    npy_std = base / "std_normalized"
    npy_patch = base / "patch_normalized"
    # Use legacy single file if this is the config default folder (backward compatibility)
    if base == BASE_DATA_PATH.resolve() and QC_RESULTS_FILE.exists():
        results_file = QC_RESULTS_FILE
    else:
        sub = _sanitize_folder_name(base)
        results_dir = QC_RESULTS_DIR / sub
        results_file = results_dir / "qc_results.json"
    return {
        "png_std_qc_path": png_std_qc,
        "png_patch_qc_path": png_patch_qc,
        "npy_std_path": npy_std,
        "npy_patch_path": npy_patch,
        "results_file": results_file,
    }


def validate_data_folder(path: Path) -> tuple[bool, str]:
    """Check that path contains expected QC structure. Return (ok, message)."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return False, "Path does not exist."
    if not path.is_dir():
        return False, "Path is not a directory."
    qc_dir = path / "std_normalized" / "qc"
    if not qc_dir.is_dir():
        return False, f"Expected subfolder 'std_normalized/qc' not found under {path}"
    try:
        n = len(list(qc_dir.glob("*.png")))
    except OSError:
        return False, "Cannot read directory."
    if n == 0:
        return False, "No PNG files found in std_normalized/qc."
    return True, f"Found {n} samples."


@st.cache_resource
def get_data_manager(_base_path: Path):
    """Get or create DataManager instance for the given base data path."""
    paths = paths_from_base(_base_path)
    dm = DataManager(
        png_std_qc_path=paths["png_std_qc_path"],
        png_patch_qc_path=paths["png_patch_qc_path"],
        npy_std_path=paths["npy_std_path"],
        npy_patch_path=paths["npy_patch_path"],
        results_file=paths["results_file"],
    )
    dm.scan_samples()
    return dm


@st.cache_resource
def get_model():
    """Load model (cached)."""
    return load_model()


def render_data_folder_screen():
    """Show data folder selection when no folder is set."""
    st.title("📁 Select Data Folder")
    st.markdown("Enter the path to the **Approved** (or equivalent) data folder that contains `std_normalized/qc` and `patch_normalized/qc`.")
    
    default = str(BASE_DATA_PATH) if BASE_DATA_PATH.exists() else ""
    path_input = st.text_input(
        "Data folder path",
        value=default,
        placeholder="/path/to/data/0ABP0TFUSH1/Approved",
        key="data_folder_input",
    )
    
    if st.button("Load", type="primary"):
        if not path_input or not path_input.strip():
            st.error("Please enter a path.")
        else:
            path = _normalize_data_folder_path(path_input)
            ok, msg = validate_data_folder(path)
            if ok:
                # Store normalized path so QC results key is always the same (Linux form)
                st.session_state[DATA_FOLDER_KEY] = path.as_posix()
                st.success(msg)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(msg)
    
    st.caption("Example: /home/user/Github/rnd_q2/data/0ABP0TFUSH1/Approved")


def render_sidebar():
    """Render sidebar with navigation and statistics."""
    base_path = get_data_folder_path()
    if base_path is None:
        return None
    
    st.sidebar.title("QC Pipeline")
    st.sidebar.caption(base_path.as_posix())
    st.sidebar.markdown("---")
    
    dm = get_data_manager(base_path)
    stats = dm.get_statistics()
    
    st.sidebar.subheader("Progress")
    
    progress = stats["progress_percent"] / 100
    st.sidebar.progress(progress, text=f"{stats['progress_percent']:.1f}% Complete")
    
    st.sidebar.markdown("**Status Breakdown:**")
    for status, count in stats["by_status"].items():
        emoji = {
            QCStatus.PENDING: "⏳",
            QCStatus.APPROVED: "✅",
            QCStatus.REJECTED: "❌",
            QCStatus.EDITED: "✏️",
        }.get(status, "")
        st.sidebar.text(f"{emoji} {status}: {count}")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Auto QC", "Manual Review", "Export"],
        key="nav",
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.sidebar.button("💾 Save Results"):
        dm.save_results()
        st.sidebar.success("Results saved!")
    
    if st.sidebar.button("📁 Change Data Folder"):
        del st.session_state[DATA_FOLDER_KEY]
        st.cache_resource.clear()
        st.rerun()
    
    return page


def render_dashboard():
    """Render dashboard page."""
    st.title("📊 QC Dashboard")
    
    dm = get_data_manager(get_data_folder_path())
    stats = dm.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", stats["total"])
    with col2:
        st.metric("Pending", stats["by_status"].get(QCStatus.PENDING, 0))
    with col3:
        approved = stats["by_status"].get(QCStatus.APPROVED, 0)
        edited = stats["by_status"].get(QCStatus.EDITED, 0)
        st.metric("Approved/Edited", approved + edited)
    with col4:
        st.metric("Rejected", stats["by_status"].get(QCStatus.REJECTED, 0))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Distribution")
        status_data = pd.DataFrame([
            {"Status": str(s), "Count": c}
            for s, c in stats["by_status"].items()
        ])
        fig = px.pie(status_data, values="Count", names="Status", hole=0.4)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Label Distribution")
        samples = dm.get_all_samples()
        label_counts = {}
        for sample in samples:
            for label in sample.raw_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if label_counts:
            label_data = pd.DataFrame([
                {"Label": k, "Count": v}
                for k, v in sorted(label_counts.items())
            ])
            fig = px.bar(label_data, x="Label", y="Count")
            st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    st.subheader("Recent Samples")
    
    samples = dm.get_all_samples()[:10]
    if samples:
        df = pd.DataFrame([
            {
                "ID": s.sample_id,
                "Raw Labels": str(s.raw_labels),
                "Status": s.status,
                "Final Labels": str(s.final_labels) if s.final_labels else "-",
            }
            for s in samples
        ])
        st.dataframe(df, width="stretch")


def render_auto_qc():
    """Render auto QC page."""
    st.title("🤖 Automatic QC")
    
    dm = get_data_manager(get_data_folder_path())
    
    st.markdown("""
    1. **Run inference on all** to get model predictions for every sample.
    2. **Approve model-negative**: approve samples where the model predicts no anomaly (max prob below threshold).
    3. **Reject model-positive**: reject samples where the model predicts at least one track with prob > 0.7.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        prob_threshold = st.slider(
            "Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=PROBABILITY_THRESHOLD,
            step=0.05,
            help="Used for match-based auto-approve and for 'model negative' (max prob < this = negative)",
        )
    with col2:
        conf_threshold = st.slider(
            "Low Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Flag as low confidence if below this threshold",
        )
    
    all_samples = dm.get_all_samples()
    pending_samples = dm.get_pending_samples()
    without_predictions = [s for s in all_samples if s.model_predictions is None]
    
    st.info(f"Total: {len(all_samples)} | Pending: {len(pending_samples)} | Without predictions: {len(without_predictions)}")
    
    # --- Run inference on all ---
    st.subheader("1. Run inference on all")
    if st.button("🚀 Run inference on all samples", type="primary"):
        if not all_samples:
            st.warning("No samples to process")
        else:
            to_run = without_predictions if without_predictions else all_samples
            try:
                model, transform = get_model()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed = errors = 0
            for i, sample in enumerate(to_run):
                status_text.text(f"Processing {sample.sample_id} ({i+1}/{len(to_run)})...")
                data = dm.load_sample_data(sample)
                if data is None:
                    errors += 1
                    continue
                try:
                    probs = run_inference(model, transform, *data)
                    dm.update_sample(sample.sample_id, model_predictions=probs.tolist())
                    processed += 1
                except Exception as e:
                    errors += 1
                progress_bar.progress((i + 1) / len(to_run))
            dm.save_results()
            status_text.text("Complete!")
            st.success(f"Inference done: {processed} updated, {errors} errors.")
            st.cache_resource.clear()
            st.rerun()
    
    # --- Batch actions by model output ---
    st.subheader("2. Batch approve / reject by model")
    
    samples_with_preds = [s for s in all_samples if s.model_predictions is not None and len(s.model_predictions) > 0]
    if not samples_with_preds:
        st.caption("Run inference on all first.")
    else:
        probs_arr = np.array([s.model_predictions for s in samples_with_preds])
        max_probs = np.max(probs_arr, axis=1)
        any_above_07 = np.any(probs_arr > 0.7, axis=1)
        
        model_negative = [s for s, mx in zip(samples_with_preds, max_probs) if mx < prob_threshold]
        model_positive = [s for s, pos in zip(samples_with_preds, any_above_07) if pos]
        
        # Only among pending
        pending_negative = [s for s in model_negative if s.status == QCStatus.PENDING or s.status == "pending"]
        pending_positive = [s for s in model_positive if s.status == QCStatus.PENDING or s.status == "pending"]
        
        st.caption(f"Pending model-negative (max prob < {prob_threshold}): {len(pending_negative)} | Pending model-positive (any track > 0.7): {len(pending_positive)}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve all model-negative", help="Approve all pending samples where max probability is below threshold (model says no anomaly)"):
                for s in pending_negative:
                    dm.update_sample(
                        s.sample_id,
                        status=QCStatus.APPROVED,
                        final_labels=[],  # model says no anomaly
                    )
                dm.save_results()
                st.success(f"Approved {len(pending_negative)} model-negative samples.")
                st.cache_resource.clear()
                st.rerun()
        with col_b:
            if st.button("❌ Reject all model-positive", help="Reject all pending samples where at least one track has probability > 0.7"):
                for s in pending_positive:
                    probs = np.array(s.model_predictions)
                    high_tracks = [f"{t:03d}" for t in np.where(probs > 0.7)[0]]
                    dm.update_sample(
                        s.sample_id,
                        status=QCStatus.REJECTED,
                        final_labels=high_tracks,
                    )
                dm.save_results()
                st.success(f"Rejected {len(pending_positive)} model-positive samples.")
                st.cache_resource.clear()
                st.rerun()
    
    # --- Original match-based Auto QC ---
    st.subheader("3. Match-based auto QC (optional)")
    st.caption("Run on pending: approve when model prediction matches raw labels and confidence is high.")
    if st.button("🚀 Run Auto QC (match raw labels)"):
        if not pending_samples:
            st.warning("No pending samples to process")
        else:
            try:
                model, transform = get_model()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            auto_approved = low_confidence = processed = errors = 0
            for i, sample in enumerate(pending_samples):
                status_text.text(f"Processing {sample.sample_id}...")
                data = dm.load_sample_data(sample)
                if data is None:
                    errors += 1
                    continue
                img_std, img_patch = data
                try:
                    probs = run_inference(model, transform, img_std, img_patch)
                    dm.update_sample(sample.sample_id, model_predictions=probs.tolist())
                    max_prob = float(np.max(probs))
                    predicted_tracks = np.where(probs > prob_threshold)[0]
                    predicted_labels = [f"{t:03d}" for t in predicted_tracks]
                    raw_set = set(sample.raw_labels)
                    pred_set = set(predicted_labels)
                    if raw_set == pred_set and max_prob > prob_threshold:
                        dm.update_sample(
                            sample.sample_id,
                            status=QCStatus.APPROVED,
                            final_labels=sample.raw_labels,
                        )
                        auto_approved += 1
                    elif max_prob < conf_threshold:
                        low_confidence += 1
                    processed += 1
                except Exception as e:
                    errors += 1
                progress_bar.progress((i + 1) / len(pending_samples))
            dm.save_results()
            status_text.text("Complete!")
            st.success(f"Processed: {processed} | Auto-approved (match): {auto_approved} | Low confidence: {low_confidence} | Errors: {errors}")
            st.cache_resource.clear()
            st.rerun()


def render_sample_viewer(sample: Sample, dm: DataManager):
    """Render sample viewer with images and predictions - compact layout."""
    
    # Row 1: Images side by side (smaller)
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("STD Normalized")
        if sample.png_std_path and Path(sample.png_std_path).exists():
            st.image(sample.png_std_path, width=350)
        else:
            st.warning("Image not found")
    
    with col2:
        st.caption("Patch Normalized")
        if sample.png_patch_path and Path(sample.png_patch_path).exists():
            st.image(sample.png_patch_path, width=350)
        else:
            st.warning("Image not found")
    
    # Row 2: Info + Predictions chart (compact)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        status_colors = {
            QCStatus.PENDING: "orange",
            QCStatus.APPROVED: "green",
            QCStatus.REJECTED: "red",
            QCStatus.EDITED: "blue",
        }
        st.markdown(f"**Raw Labels:** `{sample.raw_labels}`")
        st.markdown(f"**Status:** :{status_colors.get(sample.status, 'gray')}[{sample.status}]")
        if sample.final_labels:
            st.markdown(f"**Final:** `{sample.final_labels}`")
        
        if sample.model_predictions and len(sample.model_predictions) > 0:
            probs = np.array(sample.model_predictions)
            high_prob_tracks = np.where(probs > PROBABILITY_THRESHOLD)[0]
            if len(high_prob_tracks) > 0:
                st.markdown(f"**Model (>{PROBABILITY_THRESHOLD}):** `{[f'{t:03d}' for t in high_prob_tracks]}`")
            
            probs_formatted = [f"{p:.3f}" for p in probs]
            st.markdown(f"**Probs:** `{probs_formatted}`")
            st.caption(f"{len(probs)} values, max={probs.max():.3f}")
        else:
            st.caption("⚠️ No predictions")
    
    with col2:
        if sample.model_predictions and len(sample.model_predictions) > 0:
            probs = np.array(sample.model_predictions)
            max_prob = probs.max()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{i:03d}" for i in range(len(probs))],
                y=probs,
                marker_color=["green" if p > PROBABILITY_THRESHOLD else ("orange" if p > 0.3 else "lightgray") for p in probs],
                text=[f"{p:.2f}" if p > 0.1 else "" for p in probs],
                textposition="outside",
            ))
            fig.add_hline(y=PROBABILITY_THRESHOLD, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold ({PROBABILITY_THRESHOLD})")
            
            y_max = max(1.0, max_prob * 1.2) if max_prob > 0.01 else 1.0
            fig.update_layout(
                xaxis_title="Track",
                yaxis_title="Probability",
                yaxis_range=[0, y_max],
                height=250,
                margin=dict(l=40, r=20, t=30, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
            
            top_tracks = np.argsort(probs)[-5:][::-1]
            top_info = ", ".join([f"{t:03d}={probs[t]:.3f}" for t in top_tracks if probs[t] > 0.01])
            if top_info:
                st.caption(f"Top predictions: {top_info}")
        else:
            st.caption("No model predictions available")


def render_label_editor(sample: Sample, dm: DataManager):
    """Render label editor component - compact layout."""
    
    # Track checkbox version in session state for this sample
    version_key = f"checkbox_version_{sample.sample_id}"
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    
    # Threshold and source info in one row
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        label_threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key=f"threshold_{sample.sample_id}",
        )
    
    with col2:
        if sample.final_labels is not None:
            current_labels = set(sample.final_labels)
            st.caption("Source: saved final labels")
        elif sample.model_predictions is not None:
            probs = np.array(sample.model_predictions)
            predicted_tracks = np.where(probs > label_threshold)[0]
            current_labels = set(f"{t:03d}" for t in predicted_tracks)
            st.caption(f"Source: model predictions > {label_threshold}")
        else:
            current_labels = set(sample.raw_labels)
            st.caption("Source: raw labels from filename")
    
    with col3:
        if sample.model_predictions is not None:
            if st.button("🔄 Apply", key=f"apply_threshold_{sample.sample_id}", help="Reset checkboxes to match current threshold"):
                st.session_state[version_key] += 1
                st.rerun()
    
    # Compact checkbox grid
    st.markdown("**Select anomaly tracks:**")
    cols = st.columns(22)
    selected_labels = []
    
    version = st.session_state[version_key]
    for i in range(NUM_LABELS):
        with cols[i]:
            label = f"{i:03d}"
            checked = st.checkbox(
                str(i),
                value=label in current_labels,
                key=f"label_{sample.sample_id}_{i}_v{version}",
            )
            if checked:
                selected_labels.append(label)
    
    notes = st.text_input(
        "Notes",
        value=sample.reviewer_notes,
        key=f"notes_{sample.sample_id}",
    )
    
    return selected_labels, notes


def render_manual_review():
    """Render manual review page."""
    st.title("👁️ Manual Review")
    
    dm = get_data_manager(get_data_folder_path())
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "Approved", "Rejected", "Edited"],
            key="status_filter",
        )
    
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Sample ID", "Status", "Has Predictions"],
            key="sort_by",
        )
    
    samples = dm.get_all_samples()
    
    if status_filter != "All":
        status_map = {
            "Pending": QCStatus.PENDING,
            "Approved": QCStatus.APPROVED,
            "Rejected": QCStatus.REJECTED,
            "Edited": QCStatus.EDITED,
        }
        samples = [s for s in samples if s.status == status_map[status_filter]]
    
    if sort_by == "Sample ID":
        samples.sort(key=lambda s: s.sample_id)
    elif sort_by == "Status":
        samples.sort(key=lambda s: s.status)
    elif sort_by == "Has Predictions":
        samples.sort(key=lambda s: s.model_predictions is not None, reverse=True)
    
    st.info(f"Showing {len(samples)} samples")
    
    if not samples:
        st.warning("No samples match the current filter")
        return
    
    sample_ids = [s.sample_id for s in samples]
    
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    
    if st.session_state.current_idx >= len(samples):
        st.session_state.current_idx = 0
    
    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.rerun()
    
    with col2:
        selected_idx = st.selectbox(
            "Select Sample",
            range(len(sample_ids)),
            format_func=lambda i: f"{sample_ids[i]} ({samples[i].status})",
            index=st.session_state.current_idx,
        )
        if selected_idx != st.session_state.current_idx:
            st.session_state.current_idx = selected_idx
            st.rerun()
    
    with col3:
        if st.button("Next ➡️"):
            st.session_state.current_idx = min(
                len(samples) - 1,
                st.session_state.current_idx + 1
            )
            st.rerun()
    
    with col4:
        if st.button("⏭️ Next Pending"):
            current_idx = st.session_state.current_idx
            found_idx = None
            for i in range(current_idx + 1, len(samples)):
                if samples[i].status == QCStatus.PENDING or samples[i].status == "pending":
                    found_idx = i
                    break
            if found_idx is None:
                for i in range(0, current_idx):
                    if samples[i].status == QCStatus.PENDING or samples[i].status == "pending":
                        found_idx = i
                        break
            if found_idx is not None:
                st.session_state.current_idx = found_idx
                st.rerun()
            else:
                st.toast("No pending samples found")
    
    current_sample = samples[st.session_state.current_idx]
    
    st.markdown(f"### Sample: `{current_sample.sample_id}`")
    
    render_sample_viewer(current_sample, dm)
    
    st.markdown("---")
    
    selected_labels, notes = render_label_editor(current_sample, dm)
    
    st.markdown("---")
    
    def go_to_next_pending():
        """Find and navigate to the next pending sample."""
        current_idx = st.session_state.current_idx
        found_idx = None
        for i in range(current_idx + 1, len(samples)):
            if samples[i].status == QCStatus.PENDING or samples[i].status == "pending":
                found_idx = i
                break
        if found_idx is None:
            for i in range(0, current_idx):
                if samples[i].status == QCStatus.PENDING or samples[i].status == "pending":
                    found_idx = i
                    break
        if found_idx is not None:
            st.session_state.current_idx = found_idx
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve", type="primary", width="stretch"):
            dm.update_sample(
                current_sample.sample_id,
                status=QCStatus.APPROVED,
                final_labels=selected_labels if selected_labels else current_sample.raw_labels,
                reviewer_notes=notes,
            )
            dm.save_results()
            st.toast("Approved!")
            go_to_next_pending()
            st.rerun()
    
    with col2:
        if st.button("❌ Reject", type="secondary", width="stretch"):
            dm.update_sample(
                current_sample.sample_id,
                status=QCStatus.REJECTED,
                final_labels=selected_labels,
                reviewer_notes=notes,
            )
            dm.save_results()
            st.toast("Rejected")
            go_to_next_pending()
            st.rerun()
    
    with col3:
        if st.button("✏️ Save Edit", width="stretch"):
            dm.update_sample(
                current_sample.sample_id,
                status=QCStatus.EDITED,
                final_labels=selected_labels,
                reviewer_notes=notes,
            )
            dm.save_results()
            st.toast("Saved with edits")
            go_to_next_pending()
            st.rerun()
    
    with col4:
        if st.button("🔄 Reset to Pending", width="stretch"):
            dm.update_sample(
                current_sample.sample_id,
                status=QCStatus.PENDING,
                final_labels=None,
                reviewer_notes="",
            )
            dm.save_results()
            st.info("Reset to pending")
            st.rerun()
    
    st.markdown("---")
    st.subheader("🧠 Model Inference")
    
    samples_without_predictions = [s for s in samples if s.model_predictions is None]
    st.caption(f"{len(samples_without_predictions)} samples without predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        btn_label = "🧠 Run Model on This Sample"
        if current_sample.model_predictions is not None:
            btn_label = "🔄 Re-run Model on This Sample"
        
        if st.button(btn_label, width="stretch"):
            try:
                model, transform = get_model()
                data = dm.load_sample_data(current_sample)
                
                if data is not None:
                    img_std, img_patch = data
                    probs = run_inference(model, transform, img_std, img_patch)
                    dm.update_sample(
                        current_sample.sample_id,
                        model_predictions=probs.tolist(),
                    )
                    dm.save_results()
                    st.success("Model predictions updated!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Could not load sample data")
            except Exception as e:
                st.error(f"Error running model: {e}")
    
    with col2:
        if st.button("🚀 Run Model on All Samples", width="stretch", disabled=len(samples_without_predictions) == 0):
            try:
                model, transform = get_model()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed = 0
            errors = 0
            
            for i, sample in enumerate(samples_without_predictions):
                status_text.text(f"Processing {sample.sample_id} ({i+1}/{len(samples_without_predictions)})...")
                
                data = dm.load_sample_data(sample)
                if data is None:
                    errors += 1
                    continue
                
                img_std, img_patch = data
                
                try:
                    probs = run_inference(model, transform, img_std, img_patch)
                    dm.update_sample(
                        sample.sample_id,
                        model_predictions=probs.tolist(),
                    )
                    processed += 1
                except Exception as e:
                    st.warning(f"Error on {sample.sample_id}: {e}")
                    errors += 1
                
                progress_bar.progress((i + 1) / len(samples_without_predictions))
            
            dm.save_results()
            status_text.text("Complete!")
            st.success(f"Processed {processed} samples, {errors} errors")
            st.cache_resource.clear()
            st.rerun()


def render_export():
    """Render export page."""
    st.title("📤 Export Results")
    
    dm = get_data_manager(get_data_folder_path())
    stats = dm.get_statistics()
    
    st.markdown(f"""
    **Export Summary:**
    - Total samples: {stats['total']}
    - Approved: {stats['by_status'].get(QCStatus.APPROVED, 0)}
    - Edited: {stats['by_status'].get(QCStatus.EDITED, 0)}
    - Rejected: {stats['by_status'].get(QCStatus.REJECTED, 0)}
    - Pending: {stats['by_status'].get(QCStatus.PENDING, 0)}
    """)
    
    st.markdown("---")
    
    export_name = st.text_input(
        "Export filename",
        value="qc_results_export.csv",
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_pending = st.checkbox("Include pending samples", value=False)
    with col2:
        include_rejected = st.checkbox("Include rejected samples", value=True)
    
    if st.button("📥 Generate Export", type="primary"):
        samples = dm.get_all_samples()
        
        if not include_pending:
            samples = [s for s in samples if s.status != QCStatus.PENDING]
        if not include_rejected:
            samples = [s for s in samples if s.status != QCStatus.REJECTED]
        
        if not samples:
            st.warning("No samples to export with current filters")
            return
        
        df = pd.DataFrame([
            {
                "sample_id": s.sample_id,
                "raw_labels": str(s.raw_labels),
                "final_labels": str(s.final_labels) if s.final_labels else str(s.raw_labels),
                "status": s.status,
                "model_max_prob": max(s.model_predictions) if s.model_predictions else None,
                "reviewer_notes": s.reviewer_notes,
                "timestamp": s.timestamp,
            }
            for s in samples
        ])
        
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=export_name,
            mime="text/csv",
        )
        
        st.success(f"Ready to download {len(samples)} samples")
        
        st.markdown("---")
        st.subheader("Preview")
        st.dataframe(df.head(20), width="stretch")


def main():
    """Main app entry point."""
    if get_data_folder_path() is None:
        render_data_folder_screen()
        return
    
    page = render_sidebar()
    if page is None:
        return
    
    if page == "Dashboard":
        render_dashboard()
    elif page == "Auto QC":
        render_auto_qc()
    elif page == "Manual Review":
        render_manual_review()
    elif page == "Export":
        render_export()


if __name__ == "__main__":
    main()
