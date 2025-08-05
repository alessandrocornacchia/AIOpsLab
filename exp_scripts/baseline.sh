#!/bin/bash

# Move to root folder
cd ..

python3 automated_exp.py --run-name baseline \
    --pids \
        cpu_stress_hotel_res-localization-1 memory_stress_social_net-localization-1 \
        network_delay_hotel_res-localization-1 scale_pod_zero_social_net-localization-1 \
        \
        cpu_stress_hotel_res-detection-1 memory_stress_social_net-detection-1 \
        network_delay_hotel_res-detection-1 scale_pod_zero_social_net-detection-1 \
        \
        auth_miss_mongodb-mitigation-1 scale_pod_zero_social_net-mitigation-1 \
        \
        k8s_target_port-misconfig-analysis-3 scale_pod_zero_social_net-detection-1 \
    --agents kiara_agent \
    --models qwen3:32b mistral:instruct llama3:latest \
    --max-steps 30 \
    --runs 3
