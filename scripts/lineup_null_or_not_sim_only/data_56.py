import sys
import subprocess
import os
import glob
from PIL import Image

project_dir = subprocess.run(['Rscript', '-e', 'cat(here::here())'],
                             check=True,
                             capture_output=True,
                             text=True).stdout
if project_dir not in sys.path:
    sys.path.append(project_dir)
    from scripts.shared.keras_applications_api import keras_app_api
else:
    from scripts.shared.keras_applications_api import keras_app_api

image_dir = os.path.join(project_dir,
                         "data",
                         "lineup_null_or_not_sim_only",
                         "mixed",
                         "**",
                         "*.png")

size = 56, 56

for infile in glob.glob(image_dir, recursive=True):
    file, ext = os.path.splitext(infile)
    with Image.open(infile) as im:
        im = im.resize(size)
        final_file_name = file.replace("lineup_null_or_not_sim_only", "lineup_null_or_not_sim_only_56") + ".png"
        os.makedirs(os.path.dirname(final_file_name), exist_ok = True)
        im.save(final_file_name, "PNG")
