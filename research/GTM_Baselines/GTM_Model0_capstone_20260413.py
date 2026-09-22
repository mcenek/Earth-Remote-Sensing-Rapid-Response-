"""Frozen model definitions from the previous team. No server/auth/training entrypoint.
Source: ce6844d4410b32ae4137ada9f7d44a271df9c192:ERSRR_Website/server.py
Imported only by the explicitly requested CPU prediction command.
"""
import keras
from keras import layers, ops
input_shape = (256, 256, 5)
image_size = 256
patch_size = 8
num_patches = 1024
projection_dim = 64
num_heads = 4
transformer_units = [128, 64]
transformer_layers = 8

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        bands = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * bands,
            ),
        )
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, n_heads, t_units):
    # multi-head attention layer
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=x.shape[-1], dropout=0.1)(x, x)
    # layer normalization and skip connection
    x = layers.LayerNormalization()(x + attn)
    # MLP
    mlp_output = mlp(x, hidden_units=t_units, dropout_rate=0.1)
    # layer normalization and skip connection
    x = layers.LayerNormalization()(x + mlp_output)
    return x

def decoder_block(x, skip, filters):
    # upsample with transpose convolution
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding="same")(x)
    # add and concatenate skip connection
    skip = layers.Resizing(x.shape[1], x.shape[2])(skip)
    x = layers.Concatenate()([x, skip])
    # convolutional layers
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def create_vit_encoder_decoder():
    # ViT patches and encoding
    inputs = keras.Input(shape=input_shape)
    
    # # Skip connection to retain resolution
    # skip_conn = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    
    patches = Patches(patch_size)(inputs) # create patches
    x = PatchEncoder(num_patches, projection_dim)(patches) # encode patches
    
    # Transformer encoder
    skips = []
    for depth in range(transformer_layers):
        x = transformer_block(x, num_heads, transformer_units)
        if depth in [1, 4, 7]:
            skips.append(x)
    
    # for _ in range(transformer_layers):
    #     attn = layers.MultiHeadAttention(
    #         num_heads=num_heads,
    #         key_dim=projection_dim,
    #         dropout=0.1
    #     )(encoded_patches, encoded_patches)

    #     encoded_patches = layers.LayerNormalization()(encoded_patches + attn)
    #     mlp_output = mlp(encoded_patches, hidden_units=transformer_units, dropout_rate=0.1)
    #     encoded_patches = layers.LayerNormalization()(encoded_patches + mlp_output)

    # bridge
    h = image_size // patch_size
    x = layers.Reshape((h, h, x.shape[-1]))(x)
    for i, s in enumerate(skips):
        s = layers.Reshape((h, h, s.shape[-1]))(s)
        skips[i] = s
        
    # decoder layers
    x = decoder_block(x, skips[-1], projection_dim)
    x = decoder_block(x, skips[-2], projection_dim // 2)
    x = decoder_block(x, skips[-3], projection_dim // 4)

    # # Decoder
    # x = layers.Dense(projection_dim)(encoded_patches)
    # x = layers.Reshape((image_size // patch_size, image_size // patch_size, projection_dim))(x)

    # num_upsample_blocks = round(math.log(patch_size, 2))
    # for i in range(num_upsample_blocks):
    #     x = upsampling_block(x, (projection_dim // (2**i)))

    # skip_resized = layers.Conv2D(32, 3, padding="same", activation="relu")(skip_conn)
    # x = layers.Concatenate()([x, skip_resized])

    # x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    # x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    shared_features = x

    # Methane regression head (normalized to [0,1])
    regression_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="regression_output"
    )(shared_features)

    # Binary plume segmentation head
    mask_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="mask_output"
    )(shared_features)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "regression_output": regression_output,
            "mask_output": mask_output
        }
    )

    model.summary()
    return model
