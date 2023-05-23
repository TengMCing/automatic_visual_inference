# train a single poly plot model for poly ---------------------------------

source(here::here("scripts/shared/keras_applications_api.R"))

build_with_base_model <- function(model_name = "vgg16", 
                                  dense_nodes = c(256L),
                                  dropout_rate = 0.2,
                                  batch_normalization = TRUE) {
  # Get train and validation data
  train_val <- keras_api$flow_images_from_directory(directory = here::here("data/single_plot_reject_or_not/poly/train"),
                                                    model_name = model_name)
  train_set <- train_val[[1]]
  val_set <- train_val[[2]]
  
  # Get base model
  base_model <- keras_api$init_model(model_name = model_name, 
                                     include_top = FALSE)
  base_model$trainable <- FALSE
  
  model_input <- layer_input(keras_api$get_input_shape(model_name))
  
  model_output <- model_input %>%
    
    keras_api$preprocess_input(model_name = model_name) %>%
    
    # Use the base model
    base_model() %>%
    
    # Use a global pooling layer to flatten the tensor
    layer_global_average_pooling_2d()
  
  # Build the top layers
  for (nodes in dense_nodes) {
    model_output <- model_output %>% layer_dense(units = nodes)
    
    if (batch_normalization) model_output <- model_output %>% layer_batch_normalization() 
    
    if (dropout_rate > 0L) model_output <- model_output %>% layer_dropout(rate = dropout_rate)
    
    model_output <- model_output %>% layer_activation(activation = "relu")
  }
  
  model_output <- model_output %>% layer_dense(units = 2L, activation = "softmax")
  
  list(keras_model(inputs = model_input, outputs = model_output),
       train_set,
       val_set)
}

compile_with_learning_rate <- function(model, learning_rate = 0.001) {
  # At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs 
  # slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, 
  # located at `tf.keras.optimizers.legacy.Adam`.
  model$compile(optimizer =  keras$optimizers$legacy$Adam(learning_rate = learning_rate),
                loss = "categorical_crossentropy",
                metrics = "categorical_accuracy")
  return(model)
}


c(this_model, train_set, val_set) %<-% build_with_base_model()
compile_with_learning_rate(this_model, 0.001)

callbacks <- init_callbacks(log_dir = here::here("logs/single_plot_reject_or_not/poly"),
                            patience = 20L,
                            histogram_freq = 5L,
                            write_grads = TRUE,
                            reduce_lr_on_plateau = TRUE,
                            factor = 0.5,
                            lr_patience = 5L,
                            csv_filename = here::here("history/single_plot_reject_or_not/poly.csv"))

tensorflow::tensorboard(here::here("logs"))

fit_history <- this_model$fit(x = train_set,
                              epochs = 1000L,
                              validation_data = val_set,
                              callbacks = callbacks)

this_model$save(here::here("models/single_plot_reject_or_not/poly"))

test_set <- image_data_generator()$flow_from_directory(directory = here::here("data/single_plot_reject_or_not/poly/test"),
                                                       target_size = keras_api$get_input_shape()[1:2],
                                                       batch_size = 32L,
                                                       shuffle = FALSE,
                                                       save_to_dir = FALSE)

heter_set <- image_data_generator()$flow_from_directory(directory = here::here("data/single_plot_reject_or_not/heter/train"),
                                                        target_size = keras_api$get_input_shape()[1:2],
                                                        batch_size = 32L,
                                                        shuffle = FALSE,
                                                        save_to_dir = FALSE)

poly_pred <- this_model$predict(test_set) %>%
  as.data.frame() %>%
  pull(V2)

poly_pred_on_heter <- this_model$predict(heter_set) %>%
  as.data.frame() %>%
  pull(V2)

yardstick::conf_mat(data.frame(truth = factor(test_set$labels),
                               estimate = factor(as.integer(poly_pred > 0.5))),
                    truth = truth,
                    estimate = estimate)

yardstick::conf_mat(data.frame(truth = factor(heter_set$labels),
                               estimate = factor(as.integer(poly_pred_on_heter > 0.5))),
                    truth = truth,
                    estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(test_set$labels),
                                   estimate = factor(as.integer(poly_pred > 0.5))),
                        truth = truth,
                        estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(heter_set$labels),
                                   estimate = factor(as.integer(poly_pred_on_heter > 0.5))),
                        truth = truth,
                        estimate = estimate)

yardstick::roc_curve(data.frame(truth = factor(test_set$labels),
                                estimate = 1 - poly_pred),
                     truth = truth,
                     estimate = estimate) %>%
  autoplot()

yardstick::roc_curve(data.frame(truth = factor(heter_set$labels),
                                estimate = 1 - poly_pred_on_heter),
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
