import tensorflow as tf
from tensorflow import keras
import os
import rich
import sys
from functools import partial


class keras_app_api:

    """Keras applications APIs

    This is a wrapper of the `keras.applications` module, which intends to
    expose extra model arguments and the default input size to the user.
    """

    keras_app = keras.applications

    eff_preprocessor = keras_app.efficientnet.preprocess_input
    eff_v2_preprocessor = keras_app.efficientnet_v2.preprocess_input

    resnet_preprocessor = keras_app.resnet.preprocess_input
    resnet_v2_preprocessor = keras_app.resnet_v2.preprocess_input

    convnext_preprocessor = keras_app.convnext.preprocess_input

    app_info = {"efficient_net_b0": {"input_shape": 224,
                                     "constructor": keras_app.EfficientNetB0,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b1": {"input_shape": 240,
                                     "constructor": keras_app.EfficientNetB1,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b2": {"input_shape": 260,
                                     "constructor": keras_app.EfficientNetB2,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b3": {"input_shape": 300,
                                     "constructor": keras_app.EfficientNetB3,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b4": {"input_shape": 380,
                                     "constructor": keras_app.EfficientNetB4,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b5": {"input_shape": 456,
                                     "constructor": keras_app.EfficientNetB5,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b6": {"input_shape": 528,
                                     "constructor": keras_app.EfficientNetB6,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_b7": {"input_shape": 600,
                                     "constructor": keras_app.EfficientNetB7,
                                     "preprocessor": eff_preprocessor},
                "efficient_net_v2_b0": {"input_shape": 224,
                                        "constructor": keras_app.EfficientNetV2B0,
                                        "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_b1": {"input_shape": 240,
                                        "constructor": keras_app.EfficientNetV2B1,
                                        "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_b2": {"input_shape": 260,
                                        "constructor": keras_app.EfficientNetV2B2,
                                        "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_b3": {"input_shape": 300,
                                        "constructor": keras_app.EfficientNetV2B0,
                                        "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_s": {"input_shape": 384,
                                       "constructor": keras_app.EfficientNetV2S,
                                       "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_m": {"input_shape": 480,
                                       "constructor": keras_app.EfficientNetV2M,
                                       "preprocessor": eff_v2_preprocessor},
                "efficient_net_v2_l": {"input_shape": 480,
                                       "constructor": keras_app.EfficientNetV2L,
                                       "preprocessor": eff_v2_preprocessor},
                "xception": {"input_shape": 299,
                             "constructor": keras_app.Xception,
                             "preprocessor": keras_app.xception.preprocess_input},
                "vgg16": {"input_shape": 224,
                          "constructor": keras_app.VGG16,
                          "preprocessor": keras_app.vgg16.preprocess_input},
                "vgg19": {"input_shape": 224,
                          "constructor": keras_app.VGG19,
                          "preprocessor": keras_app.vgg19.preprocess_input},
                "resnet_50": {"input_shape": 224,
                              "constructor": keras_app.ResNet50,
                              "preprocessor": resnet_preprocessor},
                "resnet_101": {"input_shape": 224,
                               "constructor": keras_app.ResNet101,
                               "preprocessor": resnet_preprocessor},
                "resnet_152": {"input_shape": 224,
                               "constructor": keras_app.ResNet152,
                               "preprocessor": resnet_preprocessor},
                "resnet_50_v2": {"input_shape": 224,
                                 "constructor": keras_app.ResNet50V2,
                                 "preprocessor": resnet_v2_preprocessor},
                "resnet_101_v2": {"input_shape": 224,
                                  "constructor": keras_app.ResNet101V2,
                                  "preprocessor": resnet_v2_preprocessor},
                "resnet_152_v2": {"input_shape": 224,
                                  "constructor": keras_app.ResNet152V2,
                                  "preprocessor": resnet_v2_preprocessor},
                "inception_v3": {"input_shape": 299,
                                 "constructor": keras_app.InceptionV3,
                                 "preprocessor": keras_app.inception_v3.preprocess_input},
                "inception_resnet_v2": {"input_shape": 299,
                                        "constructor": keras_app.InceptionResNetV2,
                                        "preprocessor": keras_app.inception_resnet_v2.preprocess_input},
                "mobile_net": {"input_shape": 224,
                               "constructor": keras_app.MobileNet,
                               "preprocessor": keras_app.mobilenet.preprocess_input},
                "mobile_net_v2": {"input_shape": 224,
                                  "constructor": keras_app.MobileNetV2,
                                  "preprocessor": keras_app.mobilenet_v2.preprocess_input},
                "mobile_net_v3_small": {"input_shape": 224,
                                        "constructor": keras_app.MobileNetV3Small,
                                        "preprocessor": keras_app.mobilenet_v3.preprocess_input},
                "mobile_net_v3_large": {"input_shape": 224,
                                        "constructor": keras_app.MobileNetV3Large,
                                        "preprocessor": keras_app.mobilenet_v3.preprocess_input},
                "dense_net_121": {"input_shape": 224,
                                  "constructor": keras_app.DenseNet121,
                                  "preprocessor": keras_app.densenet.preprocess_input},
                "dense_net_169": {"input_shape": 224,
                                  "constructor": keras_app.DenseNet169,
                                  "preprocessor": keras_app.densenet.preprocess_input},
                "dense_net_201": {"input_shape": 224,
                                  "constructor": keras_app.DenseNet201,
                                  "preprocessor": keras_app.densenet.preprocess_input},
                "nas_net_mobile": {"input_shape": 224,
                                   "constructor": keras_app.NASNetMobile,
                                   "preprocessor": keras_app.nasnet.preprocess_input},
                "nas_net_large": {"input_shape": 331,
                                  "constructor": keras_app.NASNetMobile,
                                  "preprocessor": keras_app.nasnet.preprocess_input},
                "conv_next_tiny": {"input_shape": 224,
                                   "constructor": keras_app.ConvNeXtTiny,
                                   "preprocessor": convnext_preprocessor},
                "conv_next_small": {"input_shape": 224,
                                    "constructor": keras_app.ConvNeXtSmall,
                                    "preprocessor": convnext_preprocessor},
                "conv_next_base": {"input_shape": 224,
                                   "constructor": keras_app.ConvNeXtBase,
                                   "preprocessor": convnext_preprocessor},
                "conv_next_large": {"input_shape": 224,
                                    "constructor": keras_app.ConvNeXtLarge,
                                    "preprocessor": convnext_preprocessor},
                "conv_next_xlarge": {"input_shape": 224,
                                     "constructor": keras_app.ConvNeXtXLarge,
                                     "preprocessor": convnext_preprocessor}}

    @staticmethod
    def get_input_shape(model_name):
        """Get input shape of a keras application.

        Args:
            model_name (string): The keras application name.

        Returns:
            A tuple of 3 integers.  
        """

        width = keras_app_api.app_info[model_name]["input_shape"]
        return (width, width, 3)

    @staticmethod
    def xception_like(model_name,
                      include_top=False,
                      weights="imagenet",
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000):
        """Construct a keras model that has similar arguments to Xception.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.

        Returns:
            A tuple of 3 integers.  
        """

        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes)

    @staticmethod
    def mobile_net_like(model_name,
                        include_top=True,
                        weights="imagenet",
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000,
                        classifier_activation="softmax",
                        alpha=1.0,
                        depth_multiplier=1,
                        dropout=1e-3):
        """Construct a keras model that has similar arguments to Mobile Net.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.
            classifier_activation (str): The activation function of the top 
                layer. Ignored unless `include = True`.
            alpha (float): Width multiplier in the MobileNet paper. 
                If `alpha < 1.0`, number of filters will be decreased 
                proportionally in each layer. If `alpha > 1.0`, number of 
                filters will be increased proportionally in each layer.
            depth_multiplier (float): Resolution multiplier in the MobileNet 
                paper.
            dropout (float): Dropout rate of the top layer.

        Returns:
            A `Keras.Model` instance.  
        """

        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
            alpha=alpha,
            depth_multiplier=depth_multiplier,
            dropout=dropout)

    @staticmethod
    def mobile_net_v2_like(model_name,
                           include_top=True,
                           weights="imagenet",
                           input_tensor=None,
                           input_shape=None,
                           pooling=None,
                           classes=1000,
                           classifier_activation="softmax",
                           alpha=1.0):
        """Construct a keras model that has similar arguments to Mobile Net V2.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.
            classifier_activation (str): The activation function of the top 
                layer. Ignored unless `include = True`.
            alpha (float): Width multiplier in the MobileNet paper. 
                If `alpha < 1.0`, number of filters will be decreased 
                proportionally in each layer. If `alpha > 1.0`, number of 
                filters will be increased proportionally in each layer.

        Returns:
            A `Keras.Model` instance.  
        """
        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
            alpha=alpha)

    @staticmethod
    def mobile_net_v3_small_like(model_name,
                                 include_top=True,
                                 weights="imagenet",
                                 input_tensor=None,
                                 input_shape=None,
                                 pooling=None,
                                 classes=1000,
                                 classifier_activation="softmax",
                                 alpha=1.0,
                                 minimalistic=False,
                                 dropout_rate=0.2,
                                 include_preprocessing=True):
        """Construct a keras model that has similar arguments to Mobile Net V3.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.
            classifier_activation (str): The activation function of the top 
                layer. Ignored unless `include = True`.
            alpha (float): Width multiplier in the MobileNet paper. 
                If `alpha < 1.0`, number of filters will be decreased 
                proportionally in each layer. If `alpha > 1.0`, number of 
                filters will be increased proportionally in each layer.
            minimalistic (bool): Use minimalistic models. While these 
                models are less efficient on CPU, they are much more 
                performant on GPU/DSP.
            dropout_rate (float): Dropout rate of the top layer.
            include_preprocessing (bool):  Whether to include the 
                preprocessing layer at the bottom of the network.

        Returns:
            A `Keras.Model` instance.  
        """

        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
            alpha=alpha,
            minimalistic=minimalistic,
            dropout_rate=dropout_rate,
            include_preprocessing=include_preprocessing)

    @staticmethod
    def efficient_net_b0_like(model_name,
                              include_top=True,
                              weights="imagenet",
                              input_tensor=None,
                              input_shape=None,
                              pooling=None,
                              classes=1000,
                              classifier_activation="softmax",
                              drop_connect_rate=0.2,
                              depth_divisor=8,
                              activation="swish",
                              blocks_args="default"):
        """Construct a keras model that has similar arguments to Efficient Net.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.
            classifier_activation (str): The activation function of the top 
                layer. Ignored unless `include = True`.
            drop_connect_rate (float): Dropout rate at skip connections. 
            depth_divisor (int): A unit of network width.
            activation (str): Activation function of the network.
            block_args (str): Parameters to construct block modules.

        Returns:
            A `Keras.Model` instance.  
        """

        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
            drop_connect_rate=drop_connect_rate,
            depth_divisor=depth_divisor,
            activation=activation,
            blocks_args=blocks_args)

    @staticmethod
    def efficient_net_b0_v2_like(model_name,
                                 include_top=True,
                                 weights="imagenet",
                                 input_tensor=None,
                                 input_shape=None,
                                 pooling=None,
                                 classes=1000,
                                 classifier_activation="softmax",
                                 include_preprocessing=True):
        """Construct a keras model that has similar arguments to Efficient Net V2.

        Args:
            model_name (str): The keras application name.
            include_top (bool): Whether to include the layers at the top 
                of the network for classification.
            weights (str): One of `None` (random initialization), 
                `"imagenet"` (pre-training on ImageNet), or the path to the 
                weights file to loaded.
            input_tensor (keras tensor): Input tensor of the model.
            input_shape (int): Input shape. A length two tuple.
            pooling (str): One of `None` (4D tensor output of the last 
                convolutional block), `"avg"` (apply global average pooling to 
                get 2D tensor), `"max"` (apply global max pooling to get 
                2D tensor).
            classes (int): Number of classes (output nodes). Ignored unless
            `include_top = True`.
            classifier_activation (str): The activation function of the top 
                layer. Ignored unless `include = True`.
            drop_connect_rate (float): Dropout rate at skip connections. 
            depth_divisor (int): A unit of network width.
            activation (str): Activation function of the network.
            block_args (str): Parameters to construct block modules.

        Returns:
            A `Keras.Model` instance.  
        """

        return keras_app_api.app_info[model_name]["constructor"](
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
            include_preprocessing=include_preprocessing)

    @staticmethod
    def get_constructor(model_name):
        """Get constructer of a keras application.

        Args:
            model_name (str): The keras application name.

        Returns:
            A function to construct a `Keras.Model`.  
        """

        if (model_name in ("xception",
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
                           "dense_net_121",
                           "dense_net_169",
                           "dense_net_201",
                           "nas_net_mobile",
                           "nas_net_large")):
            return partial(keras_app_api.xception_like, model_name)

        if (model_name == "mobile_net"):
            return partial(keras_app_api.mobile_net, model_name)

        if (model_name == "mobile_net_v2"):
            return partial(keras_app_api.mobile_net_v2_like, model_name)

        if (model_name in ("mobile_net_v3_small", "mobile_net_v3_large")):
            return partial(keras_app_api.mobile_net_v3_small_like, model_name)

        if (model_name in ["efficient_net_b" + str(i) for i in range(8)]):
            return partial(keras_app_api.efficient_net_b0_like, model_name)

        if (model_name in ["efficient_net_b" + str(i) + "_v2" for i in range(8)]):
            return partial(keras_app_api.efficient_net_b0_v2_like, model_name)

        if (model_name in ["conv_next_" + i for i in ["tiny", "small", "base", "large", "xlarge"]]):
            return partial(keras_app_api.efficient_net_b0_v2_like, model_name)

    @staticmethod
    def flow_images_from_dir(directory,
                             model_name="vgg16",
                             generator=None,
                             target_size=None,
                             color_mode="rgb",
                             classes=None,
                             class_mode="categorical",
                             batch_size=32,
                             shuffle=True,
                             seed=10086,
                             save_to_dir=None,
                             save_prefix="",
                             save_format="png",
                             follow_links=False,
                             interpolation="nearest"):
        """Flow training and validation images from a folder.

        Args:
            directory (str): Path to images. The directory should contain 
                one folder for each class.
            model_name (str): The keras application name.
            generator (`keras.preprocessing.image.ImageDataGenerator` instance): Image generator. 
                By default it will set `validation_split=0.2`.
            target_size (int): Length 2 tuple indicating the image size.
            color_mode (str): Color mode of the image.
            classes (str): Could be `None` (class labels inferred from the 
                subdirectory), or a length 2 tuple indicating the class names.
            class_mode (str): One of `"categorical"`, `"binary"`, `"spare"`, 
                `"input"` or `None`.
            batch_size (int): Batch size.
            shuffle (bool): Wether to shuffle the images.
            seed (int): Seed for shuffling.
            save_to_dir (str): Path to save augmented images.
            save_prefix (str): Prefix of saved images.
            save_format (str): Format of saved images. One of `"png"`, 
                `"jpeg"`, `"bmp"`, `"pdf"`, `"ppm"`, `"gif"`, `"tif"`, `"jpg"`.
            follow_links (bool): Whether to follow symlinks inside 
                class subdirectories.
            interpolation (str): Interpolation method used to resize image. 
                One of `"nearest"`, `"bilinear"`, `"bicubic"`. 
                Optionally supported `"lanzcos"` if `PIL.__version__ > "1.1.3"`. 
                Optionally supported `"box"` and `"hamming"` if 
                `PIL.__version__ > "3.4.0"`.

        Returns:
            A length 2 tuple containing the training images generator and the
            validation images generator.  
        """

        if (generator is None):
            generator = keras.preprocessing.image.ImageDataGenerator(
                validation_split=0.2)

        if (target_size is None):
            target_size = keras_app_api.get_input_shape(model_name)[0:2]

        train_set = generator.flow_from_directory(directory=directory,
                                                  target_size=target_size,
                                                  color_mode=color_mode,
                                                  classes=classes,
                                                  class_mode=class_mode,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  seed=seed,
                                                  save_to_dir=save_to_dir,
                                                  save_prefix=save_prefix,
                                                  save_format=save_format,
                                                  follow_links=follow_links,
                                                  interpolation=interpolation,
                                                  subset="training")

        val_set = generator.flow_from_directory(directory=directory,
                                                target_size=target_size,
                                                color_mode=color_mode,
                                                classes=classes,
                                                class_mode=class_mode,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                seed=seed,
                                                save_to_dir=save_to_dir,
                                                save_prefix=save_prefix,
                                                save_format=save_format,
                                                follow_links=follow_links,
                                                interpolation=interpolation,
                                                subset="validation")

        return train_set, val_set

    @staticmethod
    def preprocess_input(x, data_format=None, model_name="vgg16"):
        """Preprocess the input to the keras application.

        Args:
            x (keras tensor): The input.
            data_format (str): Either `"channel_first"` or `"channel_last"`.
            model_name (str): One of available Keras applications. 

        Returns:
            A keras tensor.
        """

        return keras_app_api.app_info[model_name]["preprocessor"](x, data_format=data_format)

    @staticmethod
    def init_callbacks(early_stopping=True,
                       min_delta=0,
                       patience=10,
                       baseline=None,
                       restore_best_weights=False,
                       early_stopping_verbose=1,
                       tensorboard=True,
                       log_dir=None,
                       histogram_freq=0,
                       write_graph=False,
                       write_grads=False,
                       write_images=False,
                       update_freq="epoch",
                       reduce_lr_on_plateau=False,
                       factor=0.1,
                       lr_patience=10,
                       min_lr=0,
                       lr_verbose=1,
                       terminate_on_nan=False,
                       csv_logger=True,
                       csv_filename=None,
                       append=False):
        """Create a list of callbacks.

        Args:
            early_stopping (bool): Whether to enable early stopping.
            min_delta (float): Minimum change required to be considered 
                as improvement.
            patience (int): Number of epochs to wait before stopping the 
                training due to lack of improvement.
            baseline (float): Baseline value to reach in `n` (`n = patience`) 
                epochs.
            restore_best_weights (bool): Whether to restore weights from 
                the best epoch.
            early_stopping_verbose (bool): Whether to turn on verbose mode for
                early stopping.
            tensorboard (bool): Whether to enable TensorBoard.
            log_dir (str): Path to save the TensorBoard log.
            histogram_freq (int): Frequency in epochs to compute 
                activation histograms in TensorBoard.
            write_graph (bool): Whether to visualize the network in TensorBoard.
            write_grads (bool): Whether to visualize gradient histogram in 
                TensorBoard.
            write_images (bool): Whether to visualize model weights as images
                in TensorBoard.
            update_freq (str/int): `"epoch"`, `"batch"` or integer number of 
                batches to update TensorBoard log.
            reduce_lr_on_plateau (bool): Whether to reduce learning rate 
                once learning stagnates.
            factor (float): Learning rate multiplier once learning stagnates.
            lr_patience (int): Number of epochs before reducing the learning
                rate due to learning stagnates.
            min_lr (float): Lower bound of learning rate.
            lr_verbose (bool): Whether to turn on verbose mode for learning 
                rate reduction.
            terminate_on_nan (bool): Whether to terminate when encounter NAN 
                loss.
            csv_logger (bool): Whether to save training history as a csv file.
            csv_filename (str): Path to save csv file.
            append (bool): Overwrite or append to the csv file.

        Returns:
            A list of `keras.callbacks.Callback` instances.
        """

        callbacks = []

        if (early_stopping):
            callbacks.append(keras.callbacks.EarlyStopping(
                min_delta=min_delta,
                patience=patience,
                baseline=baseline,
                restore_best_weights=restore_best_weights,
                verbose=early_stopping_verbose))

        if (tensorboard):
            if os.path.exists(os.path.dirname(log_dir)) is False:
                os.mkdir(os.path.dirname(log_dir))

            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=histogram_freq,
                write_graph=write_graph,
                write_grads=write_grads,
                write_images=write_images,
                update_freq=update_freq))

        if (reduce_lr_on_plateau):
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                factor=factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=lr_verbose))

        if (terminate_on_nan):
            callbacks.append(keras.callbacks.TerminateOnNaN())

        if (csv_logger):
            if os.path.exists(os.path.dirname(csv_filename)) is False:
                os.mkdir(os.path.dirname(csv_filename))

            callbacks.append(keras.callbacks.CSVLogger(
                filename=csv_filename,
                append=append))

        return callbacks

    @staticmethod
    def api_check():
        """Check the keras API status.
        """

        rich.print("\nAPI status:")
        rich.print(f"\tUsing Python: {sys.version}")
        rich.print(f"\tUsing tensorflow: {tf.__version__}")
        rich.print(f"\tUsing keras: {keras.__version__}")
        rich.print(
            f"\tAvailable GPUs: {tf.config.list_physical_devices('GPU')}")
        rich.print("\n")


keras_app_api.api_check()
