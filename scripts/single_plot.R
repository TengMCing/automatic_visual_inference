# train single plot model -------------------------------------------------

library(tensorflow)
library(keras)

# Define image generator

data_gen <- image_data_generator(rotation_range = 5L,
                                 brightness_range = c(0.95, 1.05),
                                 validation_split = 0.2)

train_gen <- data_gen$flow_from_directory(directory = here::here("data/single_plot"),
                                          target_size = c(224L, 224L),
                                          batch_size = 32L,
                                          shuffle = TRUE,
                                          seed = 10086L,
                                          save_to_dir = FALSE,
                                          classes = c("not_reject", "reject"),
                                          subset = "training")

val_gen <- data_gen$flow_from_directory(directory = here::here("data/single_plot"),
                                        target_size = c(224L, 224L),
                                        batch_size = 32L,
                                        shuffle = TRUE,
                                        seed = 10086L,
                                        save_to_dir = FALSE,
                                        classes = c("not_reject", "reject"),
                                        subset = "validation")

# Check sample plot
reticulate::py_run_string(
  "import matplotlib.pyplot as plt; plt.imshow(next(r.train_gen)[0][0]/255); plt.show()"
  )


# Load the base model

base_model <- application_vgg16(include_top = FALSE,
                                weights = "imagenet",
                                input_tensor = NULL,
                                input_shape = c(224L, 224L, 3L),
                                pooling = NULL)


base_model$trainable <- FALSE

base_model

# Define our model

this_model <- (function() {
  # This is the input layer of our model
  model_input <- layer_input(c(reticulate::py_none(), reticulate::py_none(), 3))
  
  model_output <- model_input %>%
    
    keras$applications$vgg16$preprocess_input() %>%
    
    # Use the base model
    base_model() %>%
    
    # Use a global pooling layer to flatten the tensor
    layer_global_average_pooling_2d() %>%
    
    # Use a dense layer as the classifier
    layer_dense(units = 256L, activation = "relu") %>%
    
    # Use a dropout layer to control overfitting
    layer_dropout(rate = 0.2) %>%
    
    # This is the output layer of our model
    layer_dense(units = 2L, activation = "softmax")
  
  keras_model(inputs = model_input, outputs = model_output)
})()

# Compile our model
this_model$compile(optimizer = "adam", 
                   loss = "categorical_crossentropy",
                   metrics = "categorical_accuracy")

this_model

# To monitor the training process
this_callbacks = list(callback_early_stopping(patience = 5L), 
                      callback_tensorboard("logs/single_plot"))


fit_history <- this_model$fit(x = train_gen, 
                              epochs = 30L, 
                              validation_data = val_gen,
                              callbacks = this_callbacks)
