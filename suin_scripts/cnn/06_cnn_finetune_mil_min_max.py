#!/usr/bin/env python
"""
Thin wrapper around 06_cnn_finetune_mil.py that pins --pooling=min_max.

All actual training logic lives in 06_cnn_finetune_mil.py -- this file exists
only so job names / script names can distinguish pooling variants in file
listings and logs. If you need to change the training logic, edit
06_cnn_finetune_mil.py once; do NOT duplicate logic into this file, or the
four pooling variants will silently drift apart and stop being a fair comparison.
"""
import sys
import runpy

sys.argv += ["--pooling", "min_max"]
runpy.run_path(__file__.replace("_min_max.py", ".py"), run_name="__main__")
