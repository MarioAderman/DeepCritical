"""
DeepResearch Tools - Compatibility module for importing tools from the src directory.

This module provides backward compatibility for imports that expect tools to be at the root level.
"""

# Re-export everything from src.tools for backward compatibility
from .src.tools import *
