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
                         "single_plot_reject_or_not",
                         "mixed",
                         "train")

train_set, val_set = keras_app_api.flow_images_from_dir(directory=train_dir,
                                                        model_name="vgg16",
                                                        class_mode="categorical",
                                                        batch_size=32)
                                                        
                                                        
                                                        
def build_model(hp):
    
    # Get base model
    vgg16 = keras_app_api.get_constructor("vgg16")
    base_model = vgg16(include_top=False, weights=None)
    
    # Define the base layers
    model_input = keras.layers.Input(shape=keras_app_api.get_input_shape("vgg16"))
    model_output = keras_app_api.preprocess_input(model_input, model_name="vgg16")
    model_output = base_model(model_output)
    
    # Define the classifier
    if hp.Boolean('max_pooling'):
        model_output = keras.layers.GlobalMaxPooling2D()(model_output)
    else:
        model_output = keras.layers.GlobalAveragePooling2D()(model_output)
        
    model_output = keras.layers.Dense(
        hp.Choice('units', [128, 256, 512, 1024]),
        kernel_regularizer=keras.regularizers.L1L2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-1, step=2, sampling='log'), 
                                                   l2=hp.Float('l2', min_value=1e-5, max_value=1e-1, step=2, sampling='log')))(model_output)
    model_output = keras.layers.BatchNormalization(fused=False)(model_output)
    model_output = keras.layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.7, step=0.1))(model_output)
    model_output = keras.layers.Activation(activation="relu")(model_output)
    model_output = keras.layers.Dense(2, activation="softmax")(model_output)
    this_model = keras.Model(model_input, model_output)
    
    # Compile the model
    this_model.compile(keras.optimizers.legacy.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, step=2, sampling='log')),
                       loss="categorical_crossentropy",
                       metrics=["categorical_accuracy"])
    return this_model
  
tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                         objective='val_categorical_accuracy',
                                         max_trials=30,
                                         executions_per_trial=1,
                                         overwrite=False,
                                         directory="hyperparameter_tuning/tuner/single_plot_reject_or_not_sim_only",
                                         project_name='mixed')

# Check search space
tuner.search_space_summary()

log_dir = os.path.join(project_dir,
                       "hyperparameter_tuning",
                       "logs",
                       "single_plot_reject_or_not_sim_only",
                       "mixed")
csv_dir = os.path.join(project_dir,
                       "hyperparameter_tuning",
                       "history",
                       "single_plot_reject_or_not_sim_only",
                       "mixed.csv")
callbacks = keras_app_api.init_callbacks(log_dir=log_dir,
                                         patience=10,
                                         update_freq=20,
                                         reduce_lr_on_plateau=True,
                                         factor=0.5,
                                         lr_patience=3,
                                         csv_filename=csv_dir)

tuner.search(x=train_set, 
             epochs=200, 
             validation_data=val_set, 
             callbacks=callbacks)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
model_dir = os.path.join(project_dir,
                         "hyperparameter_tuning",
                         "models",
                         "single_plot_reject_or_not_sim_only",
                         "mixed")
best_model.save(model_dir)
