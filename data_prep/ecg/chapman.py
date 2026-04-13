#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.chapman_preprocess import main


if __name__ == "__main__":
    main()
