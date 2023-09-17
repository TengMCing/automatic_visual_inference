import sys
import subprocess
import os
import glob
from PIL import Image

project_dir = subprocess.run(['Rscript', '-e', 'cat(here::here())'],
                             check=True,
                             capture_output=True,
                             text=True).stdout
                             
data_plot_positions = subprocess.run(['Rscript', '-e', 'cat(readRDS(here::here("data/lineup_null_or_not_sim_only/meta.rds"))$k)'],
                                     check=True,
                                     capture_output=True,
                                     text=True).stdout
                                     
data_plot_positions = data_plot_positions.split(" ")

image_dir = os.path.join(project_dir,
                         "data",
                         "lineup_null_or_not_sim_only",
                         "mixed",
                         "**",
                         "*.png")
                         
size = 224, 224

for infile in glob.glob(image_dir, recursive=True):
    file, ext = os.path.splitext(infile)
    plot_uid = int(file.split("/")[-1])
    if "/null/" in file:
      continue
    with Image.open(infile) as im:
        im = im.resize(size)
        final_file_name = file.replace("lineup_null_or_not_sim_only", "lineup_position_sim_only_224")
        final_file_name = final_file_name.replace("not_null/", data_plot_positions[plot_uid - 1] + "/")
        final_file_name = final_file_name.replace("null/", data_plot_positions[plot_uid - 1] + "/")
        final_file_name = final_file_name + ".png"
        
        os.makedirs(os.path.dirname(final_file_name), exist_ok = True)
        im.save(final_file_name, "PNG")
