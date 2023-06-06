source(here::here("scripts/shared/keras_applications_api.R"))

build_with_base_model <- function(model_name = "vgg16", 
                                  dense_nodes = c(256L),
                                  dropout_rate = 0.2,
                                  batch_normalization = TRUE) {
  # Get train and validation data
  train_val <- keras_api$flow_images_from_directory(directory = here::here("data/single_plot_null_or_not_sim_only/mixed/train"),
                                                    model_name = model_name)
  train_set <- train_val[[1]]
  val_set <- train_val[[2]]
  
  # Get base model
  base_model <- keras_api$init_model(model_name = model_name, 
                                     include_top = FALSE,
                                     weights = reticulate::py_none())
  
  base_model$trainable <- TRUE

  model_input <- keras$layers$Input(keras_api$get_input_shape(model_name))
  
  model_output <- keras_api$preprocess_input(model_input, model_name = model_name)
  
  # Use the base model
  model_output <- base_model(model_output)
  
  # Use a global pooling layer to flatten the tensor
  model_output <- keras$layers$GlobalAveragePooling2D()(model_output)
  
  # Build the top layers
  for (nodes in dense_nodes) {
    model_output <- keras$layers$Dense(units = nodes)(model_output)
    
    if (batch_normalization) model_output <- keras$layers$BatchNormalization()(model_output)
    
    if (dropout_rate > 0L) model_output <- keras$layers$Dropout(rate = dropout_rate)(model_output)
    
    model_output <- keras$layers$Activation(activation = "relu")(model_output)
  }
  
  model_output <- keras$layers$Dense(units = 2L, activation = "softmax")(model_output)
  
  list(keras$Model(inputs = model_input, outputs = model_output),
       train_set,
       val_set)
}

compile_with_learning_rate <- function(model, learning_rate = 0.001) {
  # At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs 
  # slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, 
  # located at `tf.keras.optimizers.legacy.Adam`.
  model$compile(optimizer =  keras$optimizers$Adam(learning_rate = learning_rate),
                loss = "categorical_crossentropy",
                metrics = "categorical_accuracy")
  return(model)
}


c(this_model, train_set, val_set) %<-% build_with_base_model()
compile_with_learning_rate(this_model, 0.001)

callbacks <- init_callbacks(log_dir = here::here("logs/single_plot_null_or_not_sim_only/mixed"),
                            patience = 20L,
                            histogram_freq = 1L,
                            reduce_lr_on_plateau = TRUE,
                            factor = 0.5,
                            lr_patience = 5L,
                            csv_filename = here::here("history/single_plot_null_or_not_sim_only/mixed.csv"))

# tensorflow::tensorboard(here::here("logs"))

fit_history <- this_model$fit(x = train_set,
                              epochs = 10000L,
                              validation_data = val_set,
                              callbacks = callbacks)

this_model$save(here::here("models/single_plot_null_or_not_sim_only/mixed"))

test_set <- image_data_generator()$flow_from_directory(directory = here::here("data/single_plot_null_or_not_sim_only/mixed/test"),
                                                       target_size = keras_api$get_input_shape()[1:2],
                                                       batch_size = 32L,
                                                       shuffle = FALSE,
                                                       save_to_dir = FALSE)

test_set2 <- image_data_generator()$flow_from_directory(directory = here::here("data/single_plot_reject_or_not/mixed/train"),
                                                        target_size = keras_api$get_input_shape()[1:2],
                                                        batch_size = 32L,
                                                        shuffle = FALSE,
                                                        save_to_dir = FALSE)

xx <- this_model$predict(test_set) %>%
  as.data.frame() %>%
  pull(V1)

mixed_pred <- this_model$predict(test_set2) %>%
  as.data.frame() %>%
  pull(V1)


ggplot() +
  geom_histogram(aes(mixed_pred))

ggplot() +
  geom_histogram(aes(xx))

data.frame(pred = mixed_pred,
           class = test_set2$labels) %>%
  ggplot() +
  geom_histogram(aes(pred)) +
  facet_wrap(~class) +
  ggtitle("pr(not_null)")
  

yardstick::conf_mat(data.frame(truth = factor(test_set2$labels),
                               estimate = factor(as.integer(mixed_pred < 0.5))),
                    truth = truth,
                    estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(test_set2$labels),
                                   estimate = factor(as.integer(mixed_pred < 0.5))),
                        truth = truth,
                        estimate = estimate)

yardstick::roc_curve(data.frame(truth = factor(test_set2$labels),
                                estimate = mixed_pred),
                     truth = truth,
                     estimate = estimate) %>%
  autoplot()

(function() {
  user_input <- readline("Are you sure you want to stop the TensorBoard server? [y/n] ")
  if (user_input != 'y') {
    cat("Keep the TensorBoard server alive.")
    return(invisible(NULL))
  } else {
    tensorflow::tensorboard(here::here("logs"), action = "stop")
  }
})()

# train:
#   simulated null, not_null
# predict:
#   null, not_null
#   visual test reject, not_rejecct
