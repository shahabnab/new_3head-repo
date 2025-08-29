from keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf 

epsilon = 1e-5
smooth = 1



def time_freq_log_loss(y_true, y_pred, fft_len=512, alpha=0.2, eps=1e-6):
    # cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # --- safer log magnitude ---
    yt_mag = tf.sqrt(tf.square(y_true) + eps)  # smooth abs
    yp_mag = tf.sqrt(tf.square(y_pred) + eps)

    yt_log = tf.math.log(yt_mag)
    yp_log = tf.math.log(yp_mag)

    l_time = tf.reduce_mean(tf.square(yt_log - yp_log), axis=[1, 2])

    # freq domain
    yt_fft = tf.signal.rfft(y_true[..., 0], [fft_len])
    yp_fft = tf.signal.rfft(y_pred[..., 0], [fft_len])
    l_freq = tf.reduce_mean(tf.square(tf.abs(yt_fft) - tf.abs(yp_fft)), axis=1)

    return l_time + alpha * l_freq



def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.reshape(y_true, (-1,))
    y_pred_f = K.reshape(y_pred, (-1,))

    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def tp(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    return (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)

def tn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 



# Standard Tversky (positive class = 1)
def tversky(y_true, y_prob, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.cast(y_prob, tf.float32)      # y_prob must be in [0,1]
    y_true = tf.reshape(y_true, [-1])
    y_prob = tf.reshape(y_prob, [-1])

    tp = tf.reduce_sum(y_true * y_prob)
    fn = tf.reduce_sum(y_true * (1.0 - y_prob))
    fp = tf.reduce_sum((1.0 - y_true) * y_prob)

    return (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)

def tversky_loss(y_true, y_prob, alpha=0.7, beta=0.3, smooth=1e-6):
    return 1.0 - tversky(y_true, y_prob, alpha, beta, smooth)

def focal_tversky(y_true, y_prob, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    ti = tversky(y_true, y_prob, alpha, beta, smooth)
    return tf.pow(1.0 - ti, gamma)

def weighted_bce_logits(y_true, logits, w0=3.0, w1=1.0):
    y = tf.cast(y_true, tf.float32)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    w = y*w1 + (1.0 - y)*w0    # LOS (0) gets w0
    return tf.reduce_mean(w * ce)

@tf.function
def strong_aug(x, noise_std, gain_jitter):
    # x: (B, T, 1)
    b = tf.shape(x)[0]
    # amplitude/gain jitter (per-sample)
    gain = tf.random.uniform([b, 1, 1], 1.0 - gain_jitter, 1.0 + gain_jitter)
    xj = x * gain
    # additive Gaussian noise
    return xj + tf.random.normal(tf.shape(xj), stddev=noise_std)

# --- KL(p_teacher || p_student) for *binary* head using logits (preferred)
def kl_consistency_from_logits(student_logits, teacher_logits, T=1.0, weight=None):
    t = tf.cast(T, tf.float32)
    # build 2-class logits [0, z] to reuse stable softmax/log_softmax
    s2 = tf.concat([tf.zeros_like(student_logits), student_logits], axis=-1) / t
    q2 = tf.concat([tf.zeros_like(teacher_logits), tf.stop_gradient(teacher_logits)], axis=-1) / t
    q  = tf.nn.softmax(q2, axis=-1)               # teacher probs (stop-grad)
    logp = tf.nn.log_softmax(s2, axis=-1)         # student log-probs
    kl_per = tf.reduce_sum(q * (tf.math.log(tf.clip_by_value(q,1e-8,1.0)) - logp), axis=-1)
    if weight is not None:
        kl = tf.reduce_sum(weight * kl_per) / (tf.reduce_sum(weight) + 1e-8)
    else:
        kl = tf.reduce_mean(kl_per)
    return (t*t) * kl   # standard T^2 scaling
