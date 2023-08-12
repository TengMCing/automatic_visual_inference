import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras

project_dir = subprocess.run(['Rscript', '-e', 'cat(here::here())'],
                             check=True,
                             capture_output=True,
                             text=True).stdout
if project_dir not in sys.path:
    sys.path.append(project_dir)
    from scripts.shared.keras_applications_api import keras_app_api
else:
    from scripts.shared.keras_applications_api import keras_app_api

test_dir = os.path.join(project_dir,
                        "data",
                        "single_plot_null_or_not_sim_only",
                        "mixed",
                        "test")
