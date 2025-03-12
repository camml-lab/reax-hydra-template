#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py train.max_epochs=5 logger=csv

python src/train.py train.max_epochs=10 logger=csv
