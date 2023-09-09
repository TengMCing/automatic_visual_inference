import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner

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
                         "lineup_null_or_not_sim_only",
                         "mixed",
                         "train")

train_set, val_set = keras_app_api.flow_images_from_dir(directory=train_dir,
                                                        model_name="vgg16",
                                                        class_mode="categorical",
                                                        batch_size=32,
                                                        target_size=(224*2, 224*2))
                                                        
                                                        
def build_model():
    
    # Get base model
    vgg16 = keras_app_api.get_constructor("vgg16")
    base_model = vgg16(include_top=False, weights=None, input_shape=(224*2, 224*2, 3))
    
    # Define the base layers
    model_input = keras.layers.Input(shape=(224*2, 224*2, 3))
    model_output = keras_app_api.preprocess_input(model_input, model_name="vgg16")
    model_output = base_model(model_output)
    
    # Define the classifier
    model_output = keras.layers.GlobalAveragePooling2D()(model_output)
        
    model_output = keras.layers.Dense(
        256,
        kernel_regularizer=keras.regularizers.L1L2(l1=0.000128, 
                                                   l2=0.008192))(model_output)
    model_output = keras.layers.BatchNormalization(fused=False)(model_output)
    model_output = keras.layers.Dropout(0.5)(model_output)
    model_output = keras.layers.Activation(activation="relu")(model_output)
    model_output = keras.layers.Dense(2, activation="softmax")(model_output)
    this_model = keras.Model(model_input, model_output)
    
    # Compile the model
    this_model.compile(keras.optimizers.legacy.Adam(learning_rate=3.2e-05),
                       loss="categorical_crossentropy",
                       metrics=["categorical_accuracy"])
                       
    print(this_model.summary())
    return this_model

model = build_model()

log_dir = os.path.join(project_dir,
                       "hyperparameter_tuning",
                       "logs",
                       "lineup_null_or_not_sim_only",
                       "mixed_temp")
csv_dir = os.path.join(project_dir,
                       "hyperparameter_tuning",
                       "history",
                       "lineup_null_or_not_sim_only",
                       "mixed_temp.csv")
callbacks = keras_app_api.init_callbacks(log_dir=log_dir,
                                         patience=10,
                                         restore_best_weights=True,
                                         update_freq=20,
                                         reduce_lr_on_plateau=True,
                                         factor=0.5,
                                         lr_patience=3,
                                         csv_filename=csv_dir)

model.train(x=train_set, 
             epochs=200, 
             validation_data=val_set, 
             callbacks=callbacks)

model_dir = os.path.join(project_dir,
                         "hyperparameter_tuning",
                         "models",
                         "lineup_null_or_not_sim_only",
                         "mixed_temp")
model.save(model_dir)
