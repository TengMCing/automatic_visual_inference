library(tensorflow)
library(keras)
library(tidyverse)

keras_api <- new.env(parent = .GlobalEnv)

keras_api$model_name <- c("effcient_net_b0",
                          "effcient_net_b1", 
                          "effcient_net_b2",
                          "effcient_net_b3",
                          "effcient_net_b4",
                          "effcient_net_b5",
                          "effcient_net_b6",
                          "effcient_net_b7",
                          "effcient_net_v2_b0",
                          "effcient_net_v2_b1",
                          "effcient_net_v2_b2",
                          "effcient_net_v2_b3",
                          "effcient_net_v2_s",
                          "effcient_net_v2_m",
                          "effcient_net_v2_l",
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

keras_api$get_input_shape <- function(model_name = "vgg16") {
  c("effcient_net_b0" = 224L,
    "effcient_net_b1" = 240L,
    "effcient_net_b2" = 260L,
    "effcient_net_b3" = 300L,
    "effcient_net_b4" = 380L,
    "effcient_net_b5" = 456L,
    "effcient_net_b6" = 528L,
    "effcient_net_b7" = 600L,
    "effcient_net_v2_b0" = 224L,
    "effcient_net_v2_b1" = 240L,
    "effcient_net_v2_b2" = 260L,
    "effcient_net_v2_b3" = 300L,
    "effcient_net_v2_s" = 384L,
    "effcient_net_v2_m" = 480L,
    "effcient_net_v2_l" = 480L,
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
    "conv_next_xlarge" = 224L)[base_model_type] -> input_shape
  if (is.na(input_shape)) stop(glue::glue("Unmatched model name {model_name}!"))
  return(input_shape)
}

keras_api$model$xception <- function(include_top = TRUE, 
                                     weights = "imagenet", 
                                     input_tensor = reticulate::py_none(), 
                                     input_shape = reticulate::py_none(), 
                                     pooling = reticulate::py_none(), 
                                     classes = 1000L, 
                                     classifier_activation = "softmax") {
  keras$applications$Xception(include_top = include_top,
                              weights = weights,
                              input_tensor = input_tensor,
                              input_shape = input_shape,
                              pooling = pooling,
                              classes = classes,
                              classifier_activation = classifier_activation)
}

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
                       "nas_net_large" = "NASNetLarge"
                       )
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$xception)
    
    fn_args <- map(names(formals(keras_api$model$xception)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$xception))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

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

keras_api$model$mobile_net_v3_small <- function(include_top = TRUE,
                                       weights = "imagenet", 
                                       input_tensor = reticulate::py_none(), 
                                       input_shape = reticulate::py_none(), 
                                       pooling = reticulate::py_none(), 
                                       classes = 1000L, 
                                       classifier_activation = "softmax",
                                       alpha = 1.0,
                                       minimalistic = FALSE,
                                       depth_multiplier = 1L,
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

keras_api$model$mobile_net_v3_large <- function(include_top = TRUE,
                                                weights = "imagenet", 
                                                input_tensor = reticulate::py_none(), 
                                                input_shape = reticulate::py_none(), 
                                                pooling = reticulate::py_none(), 
                                                classes = 1000L, 
                                                classifier_activation = "softmax",
                                                alpha = 1.0,
                                                minimalistic = FALSE,
                                                depth_multiplier = 1L,
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

keras_api$model$effcient_net_b0 <- function(include_top = TRUE,
                                            weights = "imagenet", 
                                            input_tensor = reticulate::py_none(), 
                                            input_shape = reticulate::py_none(), 
                                            pooling = reticulate::py_none(), 
                                            classes = 1000L, 
                                            classifier_activation = "softmax",
                                            drop_connect_rate = 0.2,
                                            depth_divisor = 8L,
                                            activation = "swish",
                                            block_args = "default") {
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
                                    block_args = block_args)
}

(function() {
  register_models <- paste0("EffcientNetB", 1:7)
  names(register_models) <- paste0("effcient_net_b", 1:7)
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$effcient_net_b0)
    
    fn_args <- map(names(formals(keras_api$model$effcient_net_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$effcient_net_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

keras_api$model$effcient_net_v2_b0 <- function(include_top = TRUE,
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

(function() {
  register_models <- paste0("EffcientNetV2", c("B1", "B2", "B3", "S", "M", "L"))
  names(register_models) <- paste0("effcient_net_v2_", c("b1", "b2", "b3", "s", "m", "l"))
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$effcient_net_v2_b0)
    
    fn_args <- map(names(formals(keras_api$model$effcient_net_v2_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$effcient_net_v2_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

(function() {
  register_models <- paste0("ConvNeXt", c("Tiny", "Small", "Base", "Large", "XLarge"))
  names(register_models) <- paste0("conv_next_", c("tiny", "small", "base", "large", "xlarge"))
  
  for (this_model in names(register_models)) {
    keras_api$model[[this_model]] <- function(...) NULL
    formals(keras_api$model[[this_model]]) <- formals(keras_api$model$effcient_net_v2_b0)
    
    fn_args <- map(names(formals(keras_api$model$effcient_net_v2_b0)), ~as.symbol(.x))
    names(fn_args) <- names(formals(keras_api$model$effcient_net_v2_b0))
    fn_name <- register_models[this_model]
    
    body(keras_api$model[[this_model]]) <- substitute({do.call(keras$applications$fn_name, fn_args)})
  }
})()

