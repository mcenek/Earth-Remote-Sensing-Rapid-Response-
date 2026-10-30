"""Small spatial three-CNN segmentation model for five Sentinel-2 bands."""

from __future__ import annotations

import numpy as np


def _tf():
    import tensorflow as tf
    return tf


def _block(x, channels: int, name: str):
    tf = _tf()
    for i in range(2):
        x = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False,
                                   name=f"{name}_conv{i}")(x)
        x = tf.keras.layers.LayerNormalization(axis=-1, name=f"{name}_norm{i}")(x)
        x = tf.keras.layers.Activation("swish", name=f"{name}_act{i}")(x)
    return x


def build_branch(base_channels: int = 8, name: str = "branch"):
    """Return a U-Net whose outputs are full-resolution features and logits."""
    tf = _tf()
    if base_channels < 4:
        raise ValueError("base_channels must be at least 4")
    inp = tf.keras.Input(shape=(None, None, 5), name=f"{name}_sentinel")
    s0 = _block(inp, base_channels, f"{name}_enc0")
    x = tf.keras.layers.MaxPool2D(2, name=f"{name}_pool0")(s0)
    s1 = _block(x, base_channels * 2, f"{name}_enc1")
    x = tf.keras.layers.MaxPool2D(2, name=f"{name}_pool1")(s1)
    x = _block(x, base_channels * 4, f"{name}_bottleneck")
    x = tf.keras.layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_up1")(x)
    x = tf.keras.layers.Concatenate(name=f"{name}_skip1")([x, s1])
    x = _block(x, base_channels * 2, f"{name}_dec1")
    x = tf.keras.layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_up0")(x)
    x = tf.keras.layers.Concatenate(name=f"{name}_skip0")([x, s0])
    feature = _block(x, base_channels, f"{name}_feature")
    logits = tf.keras.layers.Conv2D(1, 1, name=f"{name}_logits")(feature)
    return tf.keras.Model(inp, [feature, logits], name=name)


def build_fusion(airborne, emit, base_channels: int = 8):
    """Trainable third CNN over frozen full-resolution features from both branches."""
    tf = _tf()
    airborne.trainable = False
    emit.trainable = False
    a = tf.keras.Input(shape=(None, None, 5), name="airborne_normalized_sentinel")
    e = tf.keras.Input(shape=(None, None, 5), name="emit_normalized_sentinel")
    fa, _ = airborne(a)
    fe, _ = emit(e)
    pa = tf.keras.layers.Conv2D(base_channels, 1, activation="swish", name="airborne_projection")(fa)
    pe = tf.keras.layers.Conv2D(base_channels, 1, activation="swish", name="emit_projection")(fe)
    x = tf.keras.layers.Concatenate(name="frozen_feature_join")([pa, pe])
    x = _block(x, base_channels * 2, "fusion_context")
    x = _block(x, base_channels, "fusion_decoder")
    logits = tf.keras.layers.Conv2D(1, 1, name="fusion_logits")(x)
    return tf.keras.Model([a, e], logits, name="fusion")


def native_probability(probability: np.ndarray, valid: np.ndarray, shape: tuple[int, int]):
    """Aggregate 20 m probabilities to exact native cells, requiring complete support."""
    h, w = probability.shape
    nh, nw = shape
    if nh <= 0 or nw <= 0 or h % nh or w % nw:
        raise ValueError("Target grid must be an integer coarsening of the input grid")
    fy, fx = h // nh, w // nw
    p = probability.reshape(nh, fy, nw, fx)
    v = valid.reshape(nh, fy, nw, fx).astype(bool)
    support = v.all(axis=(1, 3))
    out = np.where(support, np.where(v, p, 0).mean(axis=(1, 3)), np.nan)
    return out.astype(np.float32), support


def masked_bce_dice(native_prob, target, support):
    """TensorFlow masked BCE plus soft Dice over reviewed positive and negative cells."""
    tf = _tf()
    m = tf.cast(support, tf.float32)
    count = tf.reduce_sum(m)
    if float(count.numpy()) <= 0:
        raise ValueError("No reviewed target cells with complete Sentinel support")
    y = tf.where(support, target, tf.zeros_like(target))
    p = tf.clip_by_value(native_prob, 1e-6, 1 - 1e-6)
    bce = -tf.reduce_sum(m * (y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))) / count
    inter = tf.reduce_sum(m * p * y)
    dice = 1 - (2 * inter + 1) / (tf.reduce_sum(m * p) + tf.reduce_sum(m * y) + 1)
    return bce + dice
