import gymnasium 
from typing import Dict, List, Optional, Sequence

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()

class VisionNetwork(TFModelV2):
    """
    rllib's original VisionNetwork class, just slightly modified to extract "custom_model_config" from "model" config
    code from https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet.py (3.11.2021)

    Generic vision network implemented in ModelV2 API.
    An additional post-conv fully connected stack can be added and configured
    via the config keys:
    `post_fcnet_hiddens`: Dense layer sizes after the Conv2D stack.
    `post_fcnet_activation`: Activation function to use for this FC stack.
    """
 
    def __init__(self, obs_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        print("building model")
        import time
        time.sleep(15)

        model_config = model_config.get("custom_model_config") # extract the custom model config
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        super(VisionNetwork, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        print('num_outputs', num_outputs)

        activation = get_activation_fn(
            self.model_config.get("conv_activation"), framework="tf")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0,\
            "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="tf")

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        input_shape = obs_space.shape
        self.data_format = "channels_last"

        inputs = tf.keras.layers.Input(shape=input_shape, name="observations")
        last_layer = inputs
        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False

        # Build the action layers
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            # the following creation of a conv2d layer fails; this is only entered if at least 2 conv layers are configured
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="same",
                data_format="channels_last",
                name="conv{}".format(i))(last_layer)

        out_size, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            last_layer = tf.keras.layers.Conv2D(
                out_size if post_fcnet_hiddens else num_outputs,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_out")(last_layer)
            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + ([num_outputs]
                                                     if post_fcnet_hiddens else
                                                     [])
            feature_out = last_layer

            for i, out_size in enumerate(layer_sizes):
                feature_out = last_layer
                last_layer = tf.keras.layers.Dense(
                    out_size,
                    name="post_fcnet_{}".format(i),
                    activation=post_fcnet_activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv{}".format(len(filters)))(last_layer)

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                if post_fcnet_hiddens:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        post_fcnet_hiddens[0], [1, 1],
                        activation=post_fcnet_activation,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out")(last_layer)
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens[1:] +
                                                 [num_outputs]):
                        feature_out = last_layer
                        last_layer = tf.keras.layers.Dense(
                            out_size,
                            name="post_fcnet_{}".format(i + 1),
                            activation=post_fcnet_activation
                            if i < len(post_fcnet_hiddens) - 1 else None,
                            kernel_initializer=normc_initializer(1.0))(
                                last_layer)
                else:
                    feature_out = last_layer
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        num_outputs, [1, 1],
                        activation=None,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out")(last_layer)

                if last_cnn.shape[1] != 1 or last_cnn.shape[2] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, 1, "
                        "1, {} (`num_outputs`)] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the dims 1 and 2 "
                        "are both 1.".format(self.model_config["conv_filters"],
                                             self.num_outputs,
                                             list(last_cnn.shape)))

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                last_layer = tf.keras.layers.Flatten(
                    data_format="channels_last")(last_layer)

                # Add (optional) post-fc-stack after last Conv2D layer.
                for i, out_size in enumerate(post_fcnet_hiddens):
                    last_layer = tf.keras.layers.Dense(
                        out_size,
                        name="post_fcnet_{}".format(i),
                        activation=post_fcnet_activation,
                        kernel_initializer=normc_initializer(1.0))(last_layer)
                feature_out = last_layer
                self.num_outputs = last_layer.shape[1]
        logits_out = last_layer

        # Build the value layers
        if vf_share_layers:
            if not self.last_layer_is_flattened:
                feature_out = tf.keras.layers.Lambda(
                    lambda x: tf.squeeze(x, axis=[1, 2]))(feature_out)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(feature_out)
        else:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=stride
                    if isinstance(stride, (list, tuple)) else (stride, stride),
                    activation=activation,
                    padding="same",
                    data_format="channels_last",
                    name="conv_value_{}".format(i))(last_layer)
            out_size, kernel, stride = filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_value_{}".format(len(filters)))(last_layer)
            last_layer = tf.keras.layers.Conv2D(
                1, [1, 1],
                activation=None,
                padding="same",
                data_format="channels_last",
                name="conv_value_out")(last_layer)
            value_out = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)

        self.base_model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        # Explicit cast to float32 needed in eager.
        model_out, self._value_out = self.base_model(tf.cast(obs, tf.float32))
        # Our last layer is already flat.
        if self.last_layer_is_flattened:
            return model_out, state
        # Last layer is a n x [1,1] Conv2D -> Flatten.
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])



#visionnetwork with action masking
class VisionNetwork_1(TFModelV2):
    """
    modification of rllib's original VisionNetwork class;

    hardcoded stack of convolutional layers, followed by a dense layer;

    from tensorflow's tutorial on image classification:
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    """

    # param num_outputs: the number of output nodes from the actor network, i.e. the number of discrete actions
    def __init__(self, obs_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):

        super(VisionNetwork_1, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        self.data_format = "channels_last"


            #print("model config: {}".format(model_config))
            #print("model name: {}".format(name))
            #print("input_layer shape is {}".format(input_layer.shape))

        # frame stacking: self.view_requirements is modified, so the observation is a time series of images
        #self.view_requirements["prev_n_obs"] = ViewRequirement(
        #    data_col="obs",
        #    shift="-1:0",
        #    space=obs_space)

        # build the input layer
        # obs_space = obs_space['observations']
        # assert len(obs_space.shape) == 3
        input_layer = tf.keras.layers.Input(shape=obs_space['observations'].shape, name="observations") # no frame stacking
        #input_layer = tf.keras.layers.Input(shape=(2, obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]), name="observations") # with frame stacking, the first dimension indicates the number of frames
        #input_layer_reshaped = tf.keras.layers.Reshape(target_shape=(2 * obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]))(input_layer)
        rescaling_layer = tf.keras.layers.Rescaling(scale=1./255)(input_layer)
        #padding_layer = tf.keras.layers.ZeroPadding2D(rescaling_layer, padding=((0,3), (0,3)), # FIXME error: __init__() got multiple values for argument 'padding'
        #                                              data_format="channels_last")


        # for debugging

        print("constructing model VisionNetwork_1")
        print("observation space shape is {}".format(obs_space.shape))
        print("action space shape is {}".format(action_space.shape))
        print("num_outputs is {}".format(num_outputs))
        #print("model config: {}".format(model_config))
        #print("model name: {}".format(name))
        print("input_layer shape is {}".format(input_layer.shape))
        #print("padding layer shape is {}".format(padding_layer.shape))
        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        #self.last_layer_is_flattened = False

        # Build the action layers
        # 1) build the stack of convolutional layers
        conv_layer_1 = tf.keras.layers.Conv2D(
            filters=16, # number of filters; this is the dimensionality of the output space, i.e. the number of feature maps per batch item; output shape of Conv2D: batch_shape + (new_rows, new_cols, filters)
            kernel_size=[3,3], # size of the filter
            strides=2,
            activation="relu",
            padding="same", # one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
            data_format="channels_last",
            name="conv_layer_1")(rescaling_layer)
        # for debugging
        print("conv_layer_1 shape is {}".format(conv_layer_1.shape))

        bn_actor_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)

        conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,  # number of filters
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_2")(bn_actor_1)
        # for debugging
        print("conv_layer_2 shape is {}".format(conv_layer_2.shape))

        bn_actor_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)

        # conv_layer_3 config
        conv_layer_3 = tf.keras.layers.Conv2D(
            filters=64,  # number of filters
            kernel_size=[1,1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_3")(bn_actor_2)
        # for debugging
        print("conv_layer_3 shape is {}".format(conv_layer_3.shape))
        bn_actor_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)

        conv_layer_4 = tf.keras.layers.Conv2D(
            filters=128,  # number of filters
            kernel_size=[1, 1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_4")(bn_actor_3)
        # for debugging
        print("conv_layer_4 shape is {}".format(conv_layer_4.shape))

        bn_actor_4 = tf.keras.layers.BatchNormalization()(conv_layer_4)

        # flattening the output from the convolutional stack
        flat_layer = tf.keras.layers.Flatten()(bn_actor_4)
        # for debugging
        print("flat_layer shape is {}".format(flat_layer.shape))

        # dense layers; the final dense layer is configured with num_outputs nodes, these are the logits
        fc_layer_1 = tf.keras.layers.Dense(512, activation='relu')(flat_layer)

        bn_actor_5 = tf.keras.layers.BatchNormalization()(fc_layer_1)

        fc_layer_2 = tf.keras.layers.Dense(num_outputs)(bn_actor_5)
        logits_out = fc_layer_2
        # for debugging
        print("logits_out shape is {}".format(logits_out.shape))

        #***** actor network finished
        #**** build critic network

        vf_conv_layer_1 = tf.keras.layers.Conv2D(
            # input_shape=obs_space.shape, # required parameter if the conv layer is the first layer in the model
            filters=16,
            # number of filters; this is the dimensionality of the output space, i.e. the number of feature maps per batch item; output shape of Conv2D: batch_shape + (new_rows, new_cols, filters)
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            # one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
            data_format="channels_last",
            name="vf_conv_layer_1")(rescaling_layer)
        # for debugging
        print("vf_conv_layer_1 shape is {}".format(vf_conv_layer_1.shape))

        bn_critic_1 = tf.keras.layers.BatchNormalization()(vf_conv_layer_1)

        vf_conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,  # number of filters
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_2")(bn_critic_1)
        # for debugging
        print("vf_conv_layer_2 shape is {}".format(vf_conv_layer_2.shape))

        bn_critic_2 = tf.keras.layers.BatchNormalization()(vf_conv_layer_2)

        # conv_layer_3 config
        vf_conv_layer_3 = tf.keras.layers.Conv2D(
            filters=64,  # number of filters
            kernel_size=[1,1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_3")(bn_critic_2)
        # for debugging
        print("vf_conv_layer_3 shape is {}".format(vf_conv_layer_3.shape))

        bn_critic_3 = tf.keras.layers.BatchNormalization()(vf_conv_layer_3)

        # conv_layer_4 config
        vf_conv_layer_4 = tf.keras.layers.Conv2D(
            filters=128,  # number of filters
            kernel_size=[1, 1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_4")(bn_critic_3)
        # for debugging
        print("vf_conv_layer_4 shape is {}".format(vf_conv_layer_4.shape))

        bn_critic_4 = tf.keras.layers.BatchNormalization()(vf_conv_layer_4)
        """
        original code
        
         vf_last_layer = tf.keras.layers.Conv2D(
            1, [1, 1],
            activation=None,
            padding="same",
            data_format="channels_last",
            name="vf_last_layer")(vf_conv_layer_3)
        # for debugging
        print("vf_last_layer shape is {}".format(vf_last_layer.shape))

        value_out = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=[1, 2]))(vf_last_layer)
        """

        # flattening the output from the convolutional stack
        vf_flat_layer = tf.keras.layers.Flatten()(bn_critic_4)
        # for debugging
        print("vf_flat_layer shape is {}".format(vf_flat_layer.shape))

        # dense layers; the final dense layer produces the value output
        vf_fc_layer_1 = tf.keras.layers.Dense(512, activation='relu')(vf_flat_layer)
        bn_critic_5 = tf.keras.layers.BatchNormalization()(vf_fc_layer_1)
        vf_fc_layer_2 = tf.keras.layers.Dense(1)(bn_critic_5)
        value_out = vf_fc_layer_2
        # for debugging
        print("value_out shape is {}".format(value_out.shape))
        # ***** finished critic network

        self.base_model = tf.keras.Model(input_layer, [logits_out, value_out])
        # for debugging: print model summary
        self.base_model.summary()


    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        
        obs = input_dict["obs"]["observations"] # no frame stacking
        #obs = input_dict["prev_n_obs"] # with frame stacking
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        # Explicit cast to float32 needed in eager.
        model_out, self._value_out = self.base_model(tf.cast(obs, tf.float32))
        
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # turns probit action mask into logit action mask
        # Apply the logarithm to `action_mask`
        log_action_mask = tf.math.log(action_mask)

# Clamp the values of `log_action_mask` between -1e10 and FLOAT_MAX
        FLOAT_MAX = 3.4e38
        inf_mask = tf.clip_by_value(log_action_mask, -1e10, FLOAT_MAX)

        """
        *** original code: ***
         # Our last layer is already flat.
        if self.last_layer_is_flattened:
            return model_out, state
        # Last layer is a n x [1,1] Conv2D -> Flatten.
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state
        """
        return model_out + inf_mask, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])



#vision network without action masking

class VisionNetwork_0(TFModelV2):
    """
    modification of rllib's original VisionNetwork class;

    hardcoded stack of convolutional layers, followed by a dense layer;

    from tensorflow's tutorial on image classification:
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    """

    # param num_outputs: the number of output nodes from the actor network, i.e. the number of discrete actions
    def __init__(self, obs_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):

        super(VisionNetwork_0, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        self.data_format = "channels_last"

        # frame stacking: self.view_requirements is modified, so the observation is a time series of images
        #self.view_requirements["prev_n_obs"] = ViewRequirement(
        #    data_col="obs",
        #    shift="-1:0",
        #    space=obs_space)

        # build the input layer
        #assert len(obs_space.shape) == 3
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations") # no frame stacking
        #input_layer = tf.keras.layers.Input(shape=(2, obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]), name="observations") # with frame stacking, the first dimension indicates the number of frames
        #input_layer_reshaped = tf.keras.layers.Reshape(target_shape=(2 * obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]))(input_layer)
        rescaling_layer = tf.keras.layers.Rescaling(scale=1./255)(input_layer)
        #padding_layer = tf.keras.layers.ZeroPadding2D(rescaling_layer, padding=((0,3), (0,3)), # FIXME error: __init__() got multiple values for argument 'padding'
        #                                              data_format="channels_last")


        # for debugging
        print("constructing model VisionNetwork_1")
        print("observation space shape is {}".format(obs_space.shape))
        print("action space shape is {}".format(action_space.shape))
        print("num_outputs is {}".format(num_outputs))
        #print("model config: {}".format(model_config))
        #print("model name: {}".format(name))
        print("input_layer shape is {}".format(input_layer.shape))
        #print("padding layer shape is {}".format(padding_layer.shape))
        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        #self.last_layer_is_flattened = False

        # Build the action layers
        # 1) build the stack of convolutional layers
        conv_layer_1 = tf.keras.layers.Conv2D(
            filters=16, # number of filters; this is the dimensionality of the output space, i.e. the number of feature maps per batch item; output shape of Conv2D: batch_shape + (new_rows, new_cols, filters)
            kernel_size=[3,3], # size of the filter
            strides=2,
            activation="relu",
            padding="same", # one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
            data_format="channels_last",
            name="conv_layer_1")(rescaling_layer)
        # for debugging
        print("conv_layer_1 shape is {}".format(conv_layer_1.shape))

        bn_actor_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)

        conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,  # number of filters
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_2")(bn_actor_1)
        # for debugging
        print("conv_layer_2 shape is {}".format(conv_layer_2.shape))

        bn_actor_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)

        # conv_layer_3 config
        conv_layer_3 = tf.keras.layers.Conv2D(
            filters=64,  # number of filters
            kernel_size=[1,1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_3")(bn_actor_2)
        # for debugging
        print("conv_layer_3 shape is {}".format(conv_layer_3.shape))
        bn_actor_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)

        conv_layer_4 = tf.keras.layers.Conv2D(
            filters=128,  # number of filters
            kernel_size=[1, 1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_layer_4")(bn_actor_3)
        # for debugging
        print("conv_layer_4 shape is {}".format(conv_layer_4.shape))

        bn_actor_4 = tf.keras.layers.BatchNormalization()(conv_layer_4)

        # flattening the output from the convolutional stack
        flat_layer = tf.keras.layers.Flatten()(bn_actor_4)
        # for debugging
        print("flat_layer shape is {}".format(flat_layer.shape))

        # dense layers; the final dense layer is configured with num_outputs nodes, these are the logits
        fc_layer_1 = tf.keras.layers.Dense(512, activation='relu')(flat_layer)

        bn_actor_5 = tf.keras.layers.BatchNormalization()(fc_layer_1)

        fc_layer_2 = tf.keras.layers.Dense(num_outputs)(bn_actor_5)
        logits_out = fc_layer_2
        # for debugging
        print("logits_out shape is {}".format(logits_out.shape))

        #***** actor network finished
        #**** build critic network

        vf_conv_layer_1 = tf.keras.layers.Conv2D(
            # input_shape=obs_space.shape, # required parameter if the conv layer is the first layer in the model
            filters=16,
            # number of filters; this is the dimensionality of the output space, i.e. the number of feature maps per batch item; output shape of Conv2D: batch_shape + (new_rows, new_cols, filters)
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            # one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
            data_format="channels_last",
            name="vf_conv_layer_1")(rescaling_layer)
        # for debugging
        print("vf_conv_layer_1 shape is {}".format(vf_conv_layer_1.shape))

        bn_critic_1 = tf.keras.layers.BatchNormalization()(vf_conv_layer_1)

        vf_conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,  # number of filters
            kernel_size=[3, 3],  # size of the filter
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_2")(bn_critic_1)
        # for debugging
        print("vf_conv_layer_2 shape is {}".format(vf_conv_layer_2.shape))

        bn_critic_2 = tf.keras.layers.BatchNormalization()(vf_conv_layer_2)

        # conv_layer_3 config
        vf_conv_layer_3 = tf.keras.layers.Conv2D(
            filters=64,  # number of filters
            kernel_size=[1,1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_3")(bn_critic_2)
        # for debugging
        print("vf_conv_layer_3 shape is {}".format(vf_conv_layer_3.shape))

        bn_critic_3 = tf.keras.layers.BatchNormalization()(vf_conv_layer_3)

        # conv_layer_4 config
        vf_conv_layer_4 = tf.keras.layers.Conv2D(
            filters=128,  # number of filters
            kernel_size=[1, 1],  # size of the filter
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="vf_conv_layer_4")(bn_critic_3)
        # for debugging
        print("vf_conv_layer_4 shape is {}".format(vf_conv_layer_4.shape))

        bn_critic_4 = tf.keras.layers.BatchNormalization()(vf_conv_layer_4)
        """
        original code
        
         vf_last_layer = tf.keras.layers.Conv2D(
            1, [1, 1],
            activation=None,
            padding="same",
            data_format="channels_last",
            name="vf_last_layer")(vf_conv_layer_3)
        # for debugging
        print("vf_last_layer shape is {}".format(vf_last_layer.shape))

        value_out = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=[1, 2]))(vf_last_layer)
        """

        # flattening the output from the convolutional stack
        vf_flat_layer = tf.keras.layers.Flatten()(bn_critic_4)
        # for debugging
        print("vf_flat_layer shape is {}".format(vf_flat_layer.shape))

        # dense layers; the final dense layer produces the value output
        vf_fc_layer_1 = tf.keras.layers.Dense(512, activation='relu')(vf_flat_layer)
        bn_critic_5 = tf.keras.layers.BatchNormalization()(vf_fc_layer_1)
        vf_fc_layer_2 = tf.keras.layers.Dense(1)(bn_critic_5)
        value_out = vf_fc_layer_2
        # for debugging
        print("value_out shape is {}".format(value_out.shape))
        # ***** finished critic network

        self.base_model = tf.keras.Model(input_layer, [logits_out, value_out])
        # for debugging: print model summary
        self.base_model.summary()


    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"] # no frame stacking
        #obs = input_dict["prev_n_obs"] # with frame stacking
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        # Explicit cast to float32 needed in eager.
        model_out, self._value_out = self.base_model(tf.cast(obs, tf.float32))
        """
        *** original code: ***
         # Our last layer is already flat.
        if self.last_layer_is_flattened:
            return model_out, state
        # Last layer is a n x [1,1] Conv2D -> Flatten.
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state
        """
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


