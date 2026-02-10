"""
Re-export StripViT model for standalone inference usage.
"""

import os
import sys

# Allow import from the parent classification/ package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Models import StripViT  # noqa: F401
