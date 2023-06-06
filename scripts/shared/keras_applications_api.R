library(tidyverse)
library(keras)
library(tensorflow)

# keras_api ---------------------------------------------------------------

#' This is a wrapper of the Keras APIs.
keras_api <- new.env(parent = .GlobalEnv)


# model_name --------------------------------------------------------------

#' All available Keras applications
keras_api$model_name <- c("efficient_net_b0",
                          "efficient_net_b1", 
                          "efficient_net_b2",
                          "efficient_net_b3",
                          "efficient_net_b4",
                          "efficient_net_b5",
                          "efficient_net_b6",
                          "efficient_net_b7",
                          "efficient_net_v2_b0",
                          "efficient_net_v2_b1",
                          "efficient_net_v2_b2",
                          "efficient_net_v2_b3",
                          "efficient_net_v2_s",
                          "efficient_net_v2_m",
                          "efficient_net_v2_l",
                          "xception",
                          "vgg16",
                          "vgg19",
                          "resnet_50",
                          "resnet_101",
                          "resnet_152",
                          "resnet_50_v2",
                          "resnet_101_v2",
                          "resnet_152_v2",
                          "inception_v3",
                          "inception_resnet_v2",
                          "mobile_net",
                          "mobile_net_v2",
                          "mobile_net_v3_small",
                          "mobile_net_v3_large",
                          "dense_net_121",
                          "dense_net_169",
                          "dense_net_201",
                          "nas_net_mobile",
                          "nas_net_large",
                          "conv_next_tiny",
                          "conv_next_small",
                          "conv_next_base",
                          "conv_next_large",
                          "conv_next_xlarge")


# get_input_shape ---------------------------------------------------------

#' Get the default input shape of a Keras application
#' 
#' @param model_name Character. A model name.
#' @return A length 3 vector containing the input shape.
keras_api$get_input_shape <- function(model_name = "vgg16") {
  c("efficient_net_b0" = 224L,
    "efficient_net_b1" = 240L,
    "efficient_net_b2" = 260L,
    "efficient_net_b3" = 300L,
    "efficient_net_b4" = 380L,
    "efficient_net_b5" = 456L,
    "efficient_net_b6" = 528L,
    "efficient_net_b7" = 600L,
    "efficient_net_v2_b0" = 224L,
    "efficient_net_v2_b1" = 240L,
    "efficient_net_v2_b2" = 260L,
    "efficient_net_v2_b3" = 300L,
    "efficient_net_v2_s" = 384L,
    "efficient_net_v2_m" = 480L,
    "efficient_net_v2_l" = 480L,
    "xception" = 299L,
    "vgg16" = 224L,
    "vgg19" = 224L,
    "resnet_50" = 224L,
    "resnet_101" = 224L,
    "resnet_152" = 224L,
    "resnet_50_v2" = 224L,
    "resnet_101_v2" = 224L,
    "resnet_152_v2" = 224L,
    "inception_v3" = 299L,
    "inception_resnet_v2" = 299L,
    "mobile_net" = 224L,
    "mobile_net_v2" = 224L,
    "mobile_net_v3_small" = 224L,
    "mobile_net_v3_large" = 224L,
    "dense_net_121" = 224L,
    "dense_net_169" = 224L,
    "dense_net_201" = 224L,
    "nas_net_mobile" = 224L,
    "nas_net_large" = 331L,
    "conv_next_tiny" = 224L,
    "conv_next_small" = 224L,
    "conv_next_base" = 224L,
    "conv_next_large" = 224L,
    "conv_next_xlarge" = 224L)[model_name] -> input_shape
  if (is.na(input_shape)) stop(glue::glue("Unmatched model name {model_name}!"))
  return(unname(c(input_shape, input_shape, 3L)))
}


# model -------------------------------------------------------------------

#' Init a Xception model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @return A Keras model.
keras_api$model$xception <- function(include_top = TRUE, 
                                     weights = "imagenet", 
                                     input_tensor = reticulate::py_none(), 
                                     input_shape = reticulate::py_none(), 
                                     pooling = reticulate::py_none(), 
                                     classes = 1000L) {
  keras$applications$Xception(include_top = include_top,
                              weights = weights,
                              input_tensor = input_tensor,
                              input_shape = input_shape,
                              pooling = pooling,
                              classes = classes)
}

#' Init a Keras model.
#' 
#' Applications "vgg16", "vgg19", "resnet_50", "resnet_101", "resnet_152", 
#' "resnet_50_v2", "resnet_101_v2", "resnet_152_v2", "inception_v3",
#' "inception_resnet_v2", "dense_net_121", "dense_net_169", "dense_net_201",
#' "nas_net_mobile", "nas_net_large" share the same APIs as "xception".
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @return A Keras model.
(function() {
  register_models <- c("vgg16" = "VGG16", 
                       "vgg19" = "VGG19", 
                       "resnet_50" = "ResNet50", 
                       "resnet_101" = "ResNet101", 
                       "resnet_152" = "ResNet152", 
                       "resnet_50_v2" = "ResNet50V2", 
                       "resnet_101_v2" = "ResNet101V2", 
                       "resnet_152_v2" = "ResNet152V2",
                       "inception_v3" = "InceptionV3",
                       "inception_resnet_v2" = "InceptionResNetV2",
                       "dense_net_121" = "DenseNet121",
                       "dense_net_169" = "DenseNet169",
                       "dense_net_201" = "DenseNet201",
                       "nas_net_mobile" = "NASNetMobile",
                       "nas_net_large" = "NASNetLarge")
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$xception)
    
    fn_args <- map(names(formals(keras_api$model$xception)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$xception))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

#' Init a MobileNet model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param alpha Double. Width multiplier in the MobileNet paper. 
#' If `alpha < 1.0`, number of filters will be decreased proportionally 
#' in each layer. If `alpha > 1.0`, number of filters will be increased 
#' proportionally in each layer.
#' @param depth_multiplier Double. Resolution multiplier in the MobileNet paper.
#' @param dropout Double. Dropout rate of the top layer.
#' @return A Keras model.
keras_api$model$mobile_net <- function(include_top = TRUE,
                                       weights = "imagenet", 
                                       input_tensor = reticulate::py_none(), 
                                       input_shape = reticulate::py_none(), 
                                       pooling = reticulate::py_none(), 
                                       classes = 1000L, 
                                       classifier_activation = "softmax",
                                       alpha = 1.0,
                                       depth_multiplier = 1L,
                                       dropout = 1e-3) {
  keras$applications$MobileNet(include_top = include_top,
                               weights = weights,
                               input_tensor = input_tensor,
                               input_shape = input_shape,
                               pooling = pooling,
                               classes = classes,
                               classifier_activation = classifier_activation,
                               alpha = alpha,
                               depth_multiplier = depth_multiplier,
                               dropout = dropout)
}

#' Init a MobileNetV2 model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param alpha Double. Width multiplier in the MobileNet paper. 
#' If `alpha < 1.0`, number of filters will be decreased proportionally 
#' in each layer. If `alpha > 1.0`, number of filters will be increased 
#' proportionally in each layer.
keras_api$model$mobile_net_v2 <- function(include_top = TRUE,
                                          weights = "imagenet", 
                                          input_tensor = reticulate::py_none(), 
                                          input_shape = reticulate::py_none(), 
                                          pooling = reticulate::py_none(), 
                                          classes = 1000L, 
                                          classifier_activation = "softmax",
                                          alpha = 1.0) {
  keras$applications$MobileNetV2(include_top = include_top,
                                 weights = weights,
                                 input_tensor = input_tensor,
                                 input_shape = input_shape,
                                 pooling = pooling,
                                 classes = classes,
                                 classifier_activation = classifier_activation,
                                 alpha = alpha)
}

#' Init a MobileNetV3Small model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param alpha Double. Width multiplier in the MobileNet paper. 
#' If `alpha < 1.0`, number of filters will be decreased proportionally 
#' in each layer. If `alpha > 1.0`, number of filters will be increased 
#' proportionally in each layer.
#' @param minimalistic Boolean. Use minimalistic models. While these models are 
#' less efficient on CPU, they are much more performant on GPU/DSP.
#' @param dropout_rate Double. Dropout rate of the top layer.
#' @param include_preprocessing Boolean. Whether to include the preprocessing
#' layer at the bottomt of the network.
#' @return A Keras model.
keras_api$model$mobile_net_v3_small <- function(include_top = TRUE,
                                       weights = "imagenet", 
                                       input_tensor = reticulate::py_none(), 
                                       input_shape = reticulate::py_none(), 
                                       pooling = reticulate::py_none(), 
                                       classes = 1000L, 
                                       classifier_activation = "softmax",
                                       alpha = 1.0,
                                       minimalistic = FALSE,
                                       dropout_rate = 0.2,
                                       include_preprocessing = TRUE) {
  keras$applications$MobileNetV3Small(include_top = include_top,
                                      weights = weights,
                                      input_tensor = input_tensor,
                                      input_shape = input_shape,
                                      pooling = pooling,
                                      classes = classes,
                                      classifier_activation = classifier_activation,
                                      alpha = alpha,
                                      minimalistic = minimalistic,
                                      dropout_rate = dropout_rate,
                                      include_preprocessing = include_preprocessing)
}


#' Init a MobileNetV3Large model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param alpha Double. Width multiplier in the MobileNet paper. 
#' If `alpha < 1.0`, number of filters will be decreased proportionally 
#' in each layer. If `alpha > 1.0`, number of filters will be increased 
#' proportionally in each layer.
#' @param minimalistic Boolean. Use minimalistic models. While these models are 
#' less efficient on CPU, they are much more performant on GPU/DSP.
#' @param dropout_rate Double. Dropout rate of the top layer.
#' @param include_preprocessing Boolean. Whether to include the preprocessing
#' layer at the bottom of the network.
#' @return A Keras model.
keras_api$model$mobile_net_v3_large <- function(include_top = TRUE,
                                                weights = "imagenet", 
                                                input_tensor = reticulate::py_none(), 
                                                input_shape = reticulate::py_none(), 
                                                pooling = reticulate::py_none(), 
                                                classes = 1000L, 
                                                classifier_activation = "softmax",
                                                alpha = 1.0,
                                                minimalistic = FALSE,
                                                dropout_rate = 0.2,
                                                include_preprocessing = TRUE) {
  keras$applications$MobileNetV3Large(include_top = include_top,
                                      weights = weights,
                                      input_tensor = input_tensor,
                                      input_shape = input_shape,
                                      pooling = pooling,
                                      classes = classes,
                                      classifier_activation = classifier_activation,
                                      alpha = alpha,
                                      minimalistic = minimalistic,
                                      dropout_rate = dropout_rate,
                                      include_preprocessing = include_preprocessing)
}


#' Init a EfficientNetB0 model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param drop_connect_rate Double. Dropout rate at skip connections.
#' @param depth_divisor Integer. A unit of network width.
#' @param activation Character. Activation function of the network.
#' @param blocks_args Character. Parameters to construct block modules.
#' @return A Keras model.
keras_api$model$efficient_net_b0 <- function(include_top = TRUE,
                                             weights = "imagenet", 
                                             input_tensor = reticulate::py_none(), 
                                             input_shape = reticulate::py_none(), 
                                             pooling = reticulate::py_none(), 
                                             classes = 1000L, 
                                             classifier_activation = "softmax",
                                             drop_connect_rate = 0.2,
                                             depth_divisor = 8L,
                                             activation = "swish",
                                             blocks_args = "default") {
  keras$applications$EfficientNetB0(include_top = include_top,
                                    weights = weights, 
                                    input_tensor = input_tensor, 
                                    input_shape = input_shape, 
                                    pooling = pooling, 
                                    classes = classes, 
                                    classifier_activation = classifier_activation,
                                    drop_connect_rate = drop_connect_rate,
                                    depth_divisor = depth_divisor,
                                    activation = activation,
                                    blocks_args = blocks_args)
}


#' Init a Keras model.
#' 
#' Applications "efficient_net_b0" - "efficient_net_b7" share the same APIs. 
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param drop_connect_rate Double. Dropout rate at skip connections.
#' @param depth_divisor Integer. A unit of network width.
#' @param activation Character. Activation function of the network.
#' @param block_args Character. Parameters to construct block modules.
#' @return A Keras model.
(function() {
  register_models <- paste0("EfficientNetB", 1:7)
  names(register_models) <- paste0("efficient_net_b", 1:7)
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$efficient_net_b0)
    
    fn_args <- map(names(formals(keras_api$model$efficient_net_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$efficient_net_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()


#' Init a EfficientNetV2B0 model.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param include_preprocessing Boolean. Whether to include the preprocessing
#' layer at the bottom of the network.
#' @return A Keras model.
keras_api$model$efficient_net_v2_b0 <- function(include_top = TRUE,
                                                weights = "imagenet", 
                                                input_tensor = reticulate::py_none(), 
                                                input_shape = reticulate::py_none(), 
                                                pooling = reticulate::py_none(), 
                                                classes = 1000L, 
                                                classifier_activation = "softmax",
                                                include_preprocessing = TRUE) {
  keras$applications$EfficientNetV2B0(include_top = include_top,
                                      weights = weights,
                                      input_tensor = input_tensor,
                                      input_shape = input_shape,
                                      pooling = pooling,
                                      classes = classes,
                                      classifier_activation = classifier_activation,
                                      include_preprocessing = include_preprocessing)
}

#' Init a Keras model.
#' 
#' Applications "efficient_net_v2_b0" - "efficient_net_v2_b3", 
#' "efficient_net_v2_s", "efficient_net_v2_m", "efficient_net_v2_l",
#' "conv_next_tiny", "conv_next_small", "conv_next_base", "conv_next_large" and 
#' "conv_next_xlarge" share the same APIs.
#' 
#' @param include_top Boolean. Whether to include the layers at the top of the 
#' network for classification.
#' @param weights Character. One of `reticulate::py_none()` (random 
#' initialization), "imagenet" (pre-training on ImageNet), or the path to the
#' weights file to loaded.
#' @param input_tensor Keras tensor. Output of `layer_input()`. 
#' @param input_shape Integer. Length 3 integer vector indicating the input 
#' shape.
#' @param pooling Character. One of `reticulate::py_none()` (4D tensor output
#' of the last convolutional block), "avg" (apply global average pooling to 
#' get 2D tensor), "max" (apply global max pooling to get 2D tensor).
#' @param classes Integer. Number of classes (output nodes). Ignored unless
#' `include_top = TRUE`.
#' @param classifier_activation Character. The activation function of the
#' top layer. Ignored unless `include = TRUE`.
#' @param include_preprocessing Boolean. Whether to include the preprocessing
#' layer at the bottom of the network.
#' @return A Keras model.
(function() {
  register_models <- paste0("EfficientNetV2", c("B1", "B2", "B3", "S", "M", "L"))
  names(register_models) <- paste0("efficient_net_v2_", c("b1", "b2", "b3", "s", "m", "l"))
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$efficient_net_v2_b0)
    
    fn_args <- map(names(formals(keras_api$model$efficient_net_v2_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$efficient_net_v2_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

(function() {
  register_models <- paste0("ConvNeXt", c("Tiny", "Small", "Base", "Large", "XLarge"))
  names(register_models) <- paste0("conv_next_", c("tiny", "small", "base", "large", "xlarge"))
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$efficient_net_v2_b0)
    
    fn_args <- map(names(formals(keras_api$model$efficient_net_v2_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$efficient_net_v2_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()


# init_model --------------------------------------------------------------

#' Init a Keras model
#' 
#' @param model_name Character. One of available Keras applications.
#' @param ... Arguments passed to the init function.
#' @return A keras model.
keras_api$init_model <- function(model_name = "vgg16", ...) {
  if (is.null(keras_api$model[[model_name]])) stop(glue::glue("Unmatched model name {model_name}!"))
  keras_api$model[[model_name]](...)
}


# flow_images_from_directory ----------------------------------------------

#' Init an image generator 
#' 
#' @param directory Character. Path to images. The directory should contain 
#' one folder for each class.
#' @param model_name Character. One of available Keras applications.
#' @param generator Object. Output of `image_data_generator()`.
#' @param target_size Integer. Length 2 vector indicating the image size.
#' @param color_mode Character. Color mode of the image.
#' @param classes Character. Could be `reticulate::py_none()` (class labels 
#' inferred from the subdirectory), or a length 2 vector indicating the class 
#' names.
#' @param class_mode Character. One of "categorical", "binary", "spare", 
#' "input" or `reticulate::py_none()`.
#' @param batch_size Integer. Batch size.
#' @param shuffle Boolean. Wether to shuffle the images.
#' @param seed Integer. Seed for shuffling.
#' @param save_to_dir Character. Path to save augmented images.
#' @param save_prefix Character. Prefix of saved images.
#' @param save_format Character. Format of saved images. One of "png", "jpeg", 
#' "bmp", "pdf", "ppm", "gif", "tif", "jpg".
#' @param follow_links Boolean. Whether to follow symlinks inside 
#' class subdirectories.
#' @param interpolation Character. Interpolation method used to resize image.
#' One of "nearest", "bilinear", "bicubic". Optionally supported "lanzcos" if
#' `reticulate::py_run_string("import PIL; print(PIL.__version__)") > "1.1.3"`. 
#' Optionally supported "box" and "hamming" if 
#' `reticulate::py_run_string("import PIL; print(PIL.__version__)") > "3.4.0"`.
#' @param keep_aspect_ratio Boolean. Whether to resize images to a target size 
#' without aspect ratio distortion. The image is cropped in the center with 
#' target aspect ratio before resizing.
#' @return A list containing the training images generator and validation images
#' generator. 
keras_api$flow_images_from_directory <- function(directory,
                                                 model_name = "vgg16",
                                                 generator = keras$preprocessing$image$ImageDataGenerator(validation_split = 0.2),
                                                 target_size = keras_api$get_input_shape(model_name)[1:2],
                                                 color_mode = "rgb",
                                                 classes = reticulate::py_none(),
                                                 class_mode = "categorical",
                                                 batch_size = 32L,
                                                 shuffle = TRUE,
                                                 seed = 10086L,
                                                 save_to_dir = reticulate::py_none(),
                                                 save_prefix = "",
                                                 save_format = "png",
                                                 follow_links = FALSE,
                                                 interpolation = "nearest") {
  
  train_set <- generator$flow_from_directory(directory = directory,
                                             target_size = target_size,
                                             color_mode = color_mode,
                                             classes = classes,
                                             class_mode = class_mode,
                                             batch_size = batch_size,
                                             shuffle = shuffle,
                                             seed = seed,
                                             save_to_dir = save_to_dir,
                                             save_prefix = save_prefix,
                                             save_format = save_format,
                                             follow_links = follow_links,
                                             interpolation = interpolation,
                                             subset = "training")
  
  val_set <- generator$flow_from_directory(directory = directory,
                                           target_size = target_size,
                                           color_mode = color_mode,
                                           classes = classes,
                                           class_mode = class_mode,
                                           batch_size = batch_size,
                                           shuffle = shuffle,
                                           seed = seed,
                                           save_to_dir = save_to_dir,
                                           save_prefix = save_prefix,
                                           save_format = save_format,
                                           follow_links = follow_links,
                                           interpolation = interpolation,
                                           subset = "validation")
  return(list(train_set, val_set))
}




# preprocess_input --------------------------------------------------------

#' Preprocess the input before feeding into Keras applications.
#' 
#' @param x Keras tensor. The input.
#' @param data_format Character. Either "channel_first" or "channel_last".
#' @param model_name Character. One of aviailable Keras applications.
#' @return A keras tensor.
keras_api$preprocess_input <- function(x, 
                                       data_format = reticulate::py_none(), 
                                       model_name = "vgg16") {
  c("efficient_net_b0" = "efficientnet",
    "efficient_net_b1" = "efficientnet",
    "efficient_net_b2" = "efficientnet",
    "efficient_net_b3" = "efficientnet",
    "efficient_net_b4" = "efficientnet",
    "efficient_net_b5" = "efficientnet",
    "efficient_net_b6" = "efficientnet",
    "efficient_net_b7" = "efficientnet",
    "efficient_net_v2_b0" = "efficientnet_v2",
    "efficient_net_v2_b1" = "efficientnet_v2",
    "efficient_net_v2_b2" = "efficientnet_v2",
    "efficient_net_v2_b3" = "efficientnet_v2",
    "efficient_net_v2_s" = "efficientnet_v2",
    "efficient_net_v2_m" = "efficientnet_v2",
    "efficient_net_v2_l" = "efficientnet_v2",
    "xception" = "xception",
    "vgg16" = "vgg16",
    "vgg19" = "vgg19",
    "resnet_50" = "resnet",
    "resnet_101" = "resnet",
    "resnet_152" = "resnet",
    "resnet_50_v2" = "resnet_v2",
    "resnet_101_v2" = "resnet_v2",
    "resnet_152_v2" = "resnet_v2",
    "inception_v3" = "inception_v3",
    "inception_resnet_v2" = "inception_resnet_v2",
    "mobile_net" = "mobilenet",
    "mobile_net_v2" = "mobilenet_v2",
    "mobile_net_v3_small" = "mobilenet_v3",
    "mobile_net_v3_large" = "mobilenet_v3",
    "dense_net_121" = "densenet",
    "dense_net_169" = "densenet",
    "dense_net_201" = "densenet",
    "nas_net_mobile" = "nasnet",
    "nas_net_large" = "nasnet",
    "conv_next_tiny" = "convnext",
    "conv_next_small" = "convnext",
    "conv_next_base" = "convnext",
    "conv_next_large" = "convnext",
    "conv_next_xlarge" = "convnext")[model_name] -> module_name
  return(keras$applications[[module_name]]$preprocess_input(x, data_format = data_format))
}



# init_callbacks ----------------------------------------------------------

#' Init a series of callbacks
#' 
#' @param earlystopping Boolean. Whether to enable early stopping.
#' @param min_delta Double. Minimum change required.
#' @param patience Integer. Tolerance. Number of epochs.
#' @param baseline Double. Baseline value to reach.
#' @param restore_best_weights Boolean. Whether to restore weights from the
#' best epoch.
#' @param early_stopping_verbose Boolean. Whether to turn on verbose mode.
#' @param tensorboard Boolean. Whether to enable TensorBoard.
#' @param log_dir Character. Path to save the log.
#' @param histogram_freq Integer. Frequency in epochs to compute activation
#' histograms.
#' @param write_graph Boolean. Whether to visualize the network.
#' @param write_grads Boolean. Whether to visualize gradient histogram.
#' @param write_images Boolean. Whether to visualize model weights as images.
#' @param update_freq Character/Integer. "epoch", "batch" or integer number of
#' samples.
#' @param reduce_lr_on_plateau Boolean. Whether to reduce learning rate once
#' learning stagnates.
#' @param factor Double. Learning rate multiplier.
#' @param lr_patience. Integer. Tolerance. Number of epochs.
#' @param min_lr. Double. Lower bound of learning rate.
#' @param lr_verbose Boolean. Whether to turn on verbose mode.
#' @param terminate_on_nan Boolean. Whether to terminate when encounter NAN 
#' loss.
#' @param csv_logger Boolean. Whether to save training history as a csv file.
#' @param filename Character. Path to save csv file.
#' @param append Boolean. Overwrite or append to the csv file.
init_callbacks <- function(early_stopping = TRUE, 
                           min_delta = 0L,
                           patience = 10L,
                           baseline = reticulate::py_none(),
                           restore_best_weights = FALSE,
                           early_stopping_verbose = 1L,
                           tensorboard = TRUE,
                           log_dir = reticulate::py_none(),
                           histogram_freq = 0L,
                           write_graph = FALSE,
                           write_grads = FALSE,
                           write_images = FALSE,
                           update_freq = "epoch",
                           reduce_lr_on_plateau = FALSE,
                           factor = 0.1,
                           lr_patience = 10L,
                           min_lr = 0L,
                           lr_verbose = 1L,
                           terminate_on_nan = TRUE,
                           csv_Logger = TRUE,
                           csv_filename = reticulate::py_none(),
                           append = FALSE) {
  callbacks <- list()
  if (early_stopping) {
    callbacks$es <- keras$callbacks$EarlyStopping(min_delta = min_delta,
                                                  patience = patience,
                                                  baseline = baseline,
                                                  restore_best_weights = restore_best_weights,
                                                  verbose = early_stopping_verbose)
  }
  
  if (tensorboard) {
    if (!dir.exists(dirname(log_dir))) dir.create(dirname(log_dir))
    callbacks$tb <- keras$callbacks$TensorBoard(log_dir = log_dir,
                                                histogram_freq = histogram_freq,
                                                write_graph = write_graph,
                                                write_grads = write_grads,
                                                write_images = write_images,
                                                update_freq = update_freq)
  }
  
  if (reduce_lr_on_plateau) {
    callbacks$rlr <- keras$callbacks$ReduceLROnPlateau(factor = factor,
                                                       patience = lr_patience,
                                                       min_lr = min_lr,
                                                       verbose = lr_verbose)
  }
  
  if (terminate_on_nan) {
    callbacks$nan <- keras$callbacks$TerminateOnNaN()
  }
  
  if (csv_Logger) {
    if (!dir.exists(dirname(csv_filename))) dir.create(dirname(csv_filename))
    callbacks$csv <- keras$callbacks$CSVLogger(filename = csv_filename, 
                                               append = append)
  }
  
  return(unname(callbacks))
}


# trigger_tf_and_keras_module ---------------------------------------------

# A weird bug (Module proxy does not contain module name) will occur at the
# first use of any keras/tf methods. It is possibly because the module
# has not been fully loaded. The solution will be to manually trigger the
# module loading by calling any function from the module.

suppressWarnings(try(keras[["__version__"]]))

cli::cli_bullets(c(
  "{.alert-success Using}",
  "*" = "{.strong Python} {.version {reticulate::py_version()}}",
  "*" = "{.strong Python} {.pkg keras} {.version {keras[['__version__']]}}",
  "*" = "{.strong Python} {.pkg tensorflow} {.version {tf[['__version__']]}}"
))
