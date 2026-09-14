#!/usr/bin/env python
"""Standalone V7 distillation entry point with SEA-RAFT as the student."""

from __future__ import annotations

from train_raft_student_flow import main, parse_args


if __name__ == "__main__":
    args = parse_args(default_student_backend="sea_raft")
    if args.student_backend != "sea_raft":
        raise ValueError(
            "train_sea_raft_student_flow.py always trains a SEA-RAFT student; "
            "use train_raft_student_flow.py for the ProPainter backend."
        )
    main(args)
