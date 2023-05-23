#!/bin/zsh  

# This only works for homebrew miniconda TF
# Replace it with your own TensorBoard path
# Or you could try `tensorflow::tensorboard(here::here("logs"))` which may or
# may not work for your machine.
/opt/homebrew/Caskroom/miniconda/base/envs/tensorflow/bin/tensorboard --logdir logs                   
                   

