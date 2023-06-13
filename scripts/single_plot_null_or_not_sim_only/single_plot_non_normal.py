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

train_dir = os.path.join(project_dir,
                         "data",
                         "single_plot_null_or_not_sim_only",
                         "non_normal",
                         "train")

train_set, val_set = keras_app_api.flow_images_from_dir(directory=train_dir,
                                                        model_name="vgg16",
                                                        class_mode="categorical",
                                                        batch_size=32)

vgg16 = keras_app_api.get_constructor("vgg16")
base_model = vgg16(include_top=False, weights=None)

model_input = keras.layers.Input(shape=keras_app_api.get_input_shape("vgg16"))
model_output = keras_app_api.preprocess_input(model_input, model_name="vgg16")
model_output = base_model(model_output)
model_output = keras.layers.GlobalAveragePooling2D()(model_output)
model_output = keras.layers.Dense(
    512, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(model_output)
model_output = keras.layers.BatchNormalization(fused=False)(model_output)
model_output = keras.layers.Dropout(0.2)(model_output)
model_output = keras.layers.Activation(activation="relu")(model_output)
model_output = keras.layers.Dense(2, activation="softmax")(model_output)
this_model = keras.Model(model_input, model_output)

this_model.summary()

this_model.compile(keras.optimizers.legacy.Adam(learning_rate=1e-3),
                   loss="categorical_crossentropy",
                   metrics=["categorical_accuracy"])

log_dir = os.path.join(project_dir,
                       "logs",
                       "single_plot_null_or_not_sim_only",
                       "non_normal")
csv_dir = os.path.join(project_dir,
                       "history",
                       "single_plot_null_or_not_sim_only",
                       "non_normal.csv")
callbacks = keras_app_api.init_callbacks(log_dir=log_dir,
                                         patience=10,
                                         update_freq=20,
                                         reduce_lr_on_plateau=True,
                                         factor=0.5,
                                         lr_patience=3,
                                         csv_filename=csv_dir)

fit_history = this_model.fit(x=train_set,
                             epochs=1000,
                             validation_data=val_set,
                             callbacks=callbacks)

model_dir = os.path.join(project_dir,
                         "models",
                         "single_plot_null_or_not_sim_only",
                         "non_normal")
this_model.save(model_dir)
