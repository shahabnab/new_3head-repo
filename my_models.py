import tensorflow as tf
from tensorflow.keras import layers, models


from tensorflow.keras import backend as K

def build_encoder(seq_len: int,const_coef: float, latent_dim: int) -> models.Model:
            
            inp = layers.Input((seq_len ,1), name="input_signal")


            x = layers.Conv1D(int(32*const_coef) , 3, padding='same', name="c1")(inp)
            x = layers.BatchNormalization(name="bn1")(x)
            x = layers.ReLU(name="r1")(x)

            x = layers.Conv1D(int(64*const_coef), 3, strides=2, padding='same', name="c2")(x)
            x = layers.BatchNormalization(name="bn2")(x)
            x = layers.ReLU(name="r2")(x)

            x = layers.Conv1D(int(128*const_coef), 3, strides=2, padding='same', name="c3")(x)
            x = layers.BatchNormalization(name="bn3")(x)
            x = layers.ReLU(name="r3")(x)

            x = layers.GlobalAveragePooling1D(name="gap")(x)
            latent = layers.Dense(latent_dim, activation="linear", name="latent_vector")(x)
            model= models.Model(inp, latent, name="Encoder")

            return model

def build_decoder(latent_dim: int, seq_len: int,const_coef: float) -> models.Model:
        inp = layers.Input((latent_dim,), name="latent_input")

        # Project latent vector to initial shape
        x = layers.Dense(seq_len // 4 * 64, activation="relu", name="dec_dense_expand")(inp)
        x = layers.Reshape((seq_len // 4, 64), name="dec_reshape")(x)

        # Upsampling block 1
        x = layers.UpSampling1D(size=2, name="dec_upsample1")(x)
        x = layers.Conv1D(int(64*const_coef), 3, padding='same', name="dec_conv1")(x)
        x = layers.BatchNormalization(name="dec_bn1")(x)
        x = layers.ReLU(name="dec_r1")(x)

        # Upsampling block 2
        x = layers.UpSampling1D(size=2, name="dec_upsample2")(x)
        x = layers.Conv1D(int(32*const_coef), 3, padding='same', name="dec_conv2")(x)
        x = layers.BatchNormalization(name="dec_bn2")(x)
        x = layers.ReLU(name="dec_r2")(x)

        # Final convolution to produce output shape (seq_len, 1)
        x = layers.Conv1D(1, 3, padding='same', activation='sigmoid', name="dec_output_signal")(x)

        # Ensure correct output length if seq_len is not divisible by 4
        x = layers.Cropping1D((0, x.shape[1] - seq_len))(x) if x.shape[1] is not None and x.shape[1] > seq_len else x
        x = layers.ZeroPadding1D((0, seq_len - x.shape[1]))(x) if x.shape[1] is not None and x.shape[1] < seq_len else x

        return models.Model(inp, x, name="reconstruction")    



# ── 1) a GRL layer that holds a mutable `hp_lambda` variable ─────────────────
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, name=None, initial_lambda=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        # this is the λ we’ll schedule
        self.hp_lambda = K.variable(initial_lambda, dtype="float32", name=f"{self.name}_lambda")

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                lam_cast = tf.cast(self.hp_lambda, dy.dtype)
                return -lam_cast * dy
            return x, grad
        return _reverse(x)
def build_DA_AE(seq_len: int, n_domains: int, latent_dim: int,ENC_CONST_COEF:float, DEC_CONST_COEF:float) :
       

        encoder=build_encoder(seq_len,ENC_CONST_COEF, latent_dim,)  # Ensure encoder is built
        latent=encoder.output

        # ── Decoder (nested) ───────────────────────────────────────────────────────
        # call your existing build_decoder here

        decoder_model = build_decoder(latent_dim,seq_len,DEC_CONST_COEF)
        # this nested model must be named "reconstruction" inside build_decoder
        recon = decoder_model(latent)

        # ── Heads ──────────────────────────────────────────────────────────────────
        grl = GradientReversal(name="grl", initial_lambda=0.0)
        dom_prob  = layers.Dense(n_domains, name="domain_prob",activation="softmax")(grl(latent))

        latent_cls = layers.Dense(64, activation="relu",name="latent_cls")(latent)
        latent_cls = layers.LayerNormalization()(latent_cls)

        
        s = layers.Dense(16, activation="relu",name="excite_down")(latent_cls)          # 64 → 16
        s = layers.Dense(64, activation="sigmoid",name="excite_up")(s)                # 16 → 64

        
        gated = layers.Multiply(name="attention_gated")([latent_cls, s]) 

        los_logit = layers.Dense(1, name="los_logit")(gated)
        los_prob  = layers.Activation("sigmoid", name="los_prob")(los_logit)
        

        

        ae = models.Model(
            inputs  = encoder.input,
            outputs = [recon, dom_prob, los_prob],
            name    = "DA_AE_with_nested_decoder"
        )
        return ae, grl


