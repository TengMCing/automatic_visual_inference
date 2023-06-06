#!/bin/bash
cd sk54/patrickli/automatic_visual_inference
module load R/4.0.5
module load cuda/10.1
module load cudnn/7.6.5.32-cuda10
conda activate tf2-gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
