#!/bin/bash

export ROOT_PATH=/home/rishihazra/PycharmProjects/AutonomousDriving
# Run the script in dummy mode
python3 ui_hf.py --dummy_run --gen_id 0

python3 ui_hf.py --gen_id 0

# Uncomment the line below to run in actual mode
# python3 /path/to/your_script.py
