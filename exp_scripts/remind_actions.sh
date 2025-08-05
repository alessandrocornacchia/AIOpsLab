#!/bin/bash

# Move to root folder
cd ..

# Set ENV variable controlling the refresh of actions
python3 automated_exp.py --run-name remind_actions \
    --pids \
        cpu_stress_hotel_res-localization-1 memory_stress_social_net-localization-1 \
        network_delay_hotel_res-localization-1 scale_pod_zero_social_net-localization-1 \
    --agents kiara_agent \
    --models qwen3:32b mistral:instruct llama3:latest \
    --max-steps 30 \
    --runs 3
