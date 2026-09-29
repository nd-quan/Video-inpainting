#!/usr/bin/env python
"""Stage 1 flow validation entrypoint (EPE and warp diagnostics)."""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
runpy.run_path(str(HERE.parent / "STC_encoder_v7_raft_flow_distillation" / "train_raft_student_flow.py"), run_name="__main__")
