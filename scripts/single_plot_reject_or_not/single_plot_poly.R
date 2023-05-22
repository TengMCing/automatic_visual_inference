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




# Compile our model
this_model$compile(optimizer = "adam", 
                   loss = "categorical_crossentropy",
                   metrics = "categorical_accuracy")

this_model

# To monitor the training process
this_callbacks = list(callback_early_stopping(patience = 10L), 
                      callback_tensorboard("logs/single_plot_poly"))


fit_history <- this_model$fit(x = train_gen, 
                              epochs = 100L, 
                              validation_data = val_gen,
                              callbacks = this_callbacks)

if (!dir.exists(here::here("history"))) dir.create(here::here("history"))
saveRDS(fit_history, here::here("history/single_plot_poly.rds"))
this_model$save(here::here("models/single_plot_poly"))
