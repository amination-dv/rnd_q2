#!/bin/bash
# Quick start script for QC Pipeline

cd "$(dirname "$0")"

echo "Starting QC Pipeline..."
echo "Open http://localhost:8501 in your browser"
echo ""

/home/zmirikha/ilipy/bin/streamlit run app.py --server.port 8501
#./run.sh