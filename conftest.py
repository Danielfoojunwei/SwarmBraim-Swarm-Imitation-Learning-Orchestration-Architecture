"""Root conftest.py to configure pytest to ignore problematic files."""

import sys
import types
from pathlib import Path

# Tell pytest to ignore the root __init__.py which has relative imports
# that only work when properly installed as a package
collect_ignore = ["__init__.py"]

# Pre-register a dummy package to prevent import errors
project_root = Path(__file__).parent
project_name = project_root.name

if project_name not in sys.modules:
    sys.modules[project_name] = types.ModuleType(project_name)

# Add the project root to the path so tests can import modules directly
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
