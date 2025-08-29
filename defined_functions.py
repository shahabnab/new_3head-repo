
import numpy as np
import tensorflow as tf
import math
import time
from sklearn.model_selection import train_test_split
from my_callbacks import *
from my_losses import *
from my_plotters import *
from my_cir_processing import *
from my_df_processing import *  
from my_models import *
from my_helping_functions import *
from tensorflow.keras.losses import BinaryFocalCrossentropy
import pandas as pd
import matplotlib.pyplot as plt  # put this at the top of your file
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os

DEBUG=False


def mean_entropy(logits, *, base='e'):
    """
    Mean Shannon entropy from *binary* logits.
    Expects logits of shape (N,) or (N,1). Returns a scalar tf.Tensor.
    Uses natural log (nats) by default; set base=2 for bits.
    """
    x = tf.convert_to_tensor(logits, tf.float32)
    x = tf.reshape(x, (-1, 1))  # ensure (N,1)

    # Build 2-class logits [0, z] and compute softmax probs
    two_class = tf.concat([tf.zeros_like(x), x], axis=-1)  # (N,2)
    p = tf.nn.softmax(two_class, axis=-1)

    # PyTorch-style entropy: -mean(sum p * log(p + 1e-8))
    eps = tf.constant(1e-8, tf.float32)
    H = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)  # (N,)
    Hm = tf.reduce_mean(H)                                 # scalar

    if base == 2:
        Hm = Hm / tf.math.log(tf.constant(2.0, tf.float32))

    return float(Hm.numpy())


    
def binary_entropy(p, eps=1e-8):
    # Cast safely even if p is already a Tensor
    p = tf.cast(p, tf.float32)
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    ent = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))
    return float(tf.reduce_mean(ent).numpy())

def cons_lambda(epoch,h):
    if h["CONS_RAMP_EPOCHS"] <= 0:
        return tf.constant(h["CONS_LAMBDA"], tf.float32)
    p = tf.cast(tf.minimum(epoch, h["CONS_RAMP_EPOCHS"]), tf.float32) / float(h["CONS_RAMP_EPOCHS"])
    return tf.cast(h["CONS_LAMBDA"], tf.float32) * p




def _per_sample_mean(t):
    t = tf.convert_to_tensor(t)
    if t.shape.rank in (0, 1):    # scalar or (batch,)
        return tf.reshape(t, (-1,))
    flat = tf.reshape(t, (tf.shape(t)[0], -1))
    return tf.reduce_mean(flat, axis=1)

def _weighted_mean(per_sample, sw):
    per_sample = tf.reshape(tf.cast(per_sample, tf.float32), (-1,))
    sw = tf.reshape(tf.cast(sw, tf.float32), (-1,))
    return tf.reduce_sum(per_sample * sw) / (tf.reduce_sum(sw) + 1e-8)





# ======================================================================
    # Closed-form scores for reporting (keep names clear; probs vs logits)
    # ======================================================================
def _binary_crossentropy_mean(y_true, p, eps=1e-7):
        p = np.clip(p, eps, 1. - eps)
        ce = -(y_true*np.log(p) + (1. - y_true)*np.log(1. - p))
        return float(np.mean(ce))

def _binary_focal_mean(y_true, p, gamma=2.0, alpha=0.25, eps=1e-7):
        p = np.clip(p, eps, 1. - eps)
        pt = np.where(y_true == 1, p, 1. - p)
        alpha_t = np.where(y_true == 1, alpha, 1. - alpha)
        loss = -alpha_t * ((1. - pt) ** gamma) * np.log(pt)
        return float(np.mean(loss))







    
def train_ae(balanced_dsets, CIRS,RNG, Domains, Weights, LosLabels, h, trial=None):
    # ---- hyperparams ----
   



    # ---- assemble arrays ----
    X  = np.concatenate([CIRS[k] for k in ("TRAIN1","TRAIN2","ADAPTION")])[..., None].astype('float32', copy=True)
    d  = np.concatenate([Domains[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.int32, copy=True)
    w  = np.concatenate([Weights[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.float32,copy=True)
    l  = np.concatenate([LosLabels[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.float32,copy=True)
  


    X_test = np.array(CIRS["TEST"])[..., None].astype('float32')
    d_test = np.array(Domains["TEST"]).astype(np.int32,copy=True)
    l_test = np.array(LosLabels["TEST"]).astype(np.float32,copy=True).reshape(-1, 1)
    w_test = np.array(Weights["TEST"]).astype(np.float32,copy=True)

    num_dom = int(d.max() + 1)
    #spliting the train and validation 
    joint = list(zip(d, l))
    X_tr, X_val, d_tr, d_val, w_tr, w_val, l_tr, l_val = train_test_split(
        X, d, w, l, test_size=0.3, random_state=h["SEED"], stratify=joint, shuffle=True
    ) 
    if DEBUG:
        print("Train LOS/NLOS:", np.bincount(l_tr.astype(np.int32)))
        print("Val   LOS/NLOS:", np.bincount(l_val.astype(np.int32)))
        print("Train domain:", np.bincount(d_tr.astype(np.int32), minlength=num_dom))
        print("Val   domain:", np.bincount(d_val.astype(np.int32), minlength=num_dom))
      
        
    # ---- datasets ----
    #making the datasets
    train_ds = make_dataset(X_tr, d_tr, l_tr, w_tr, h["AE_BATCH"], num_dom,
                        seed=h["SEED"], split="train", pl_mode="hide", pl_domain_id=h["PL_DOMAIN_ID"])

    val_ds   = make_dataset(X_val, d_val, l_val, w_val, h["AE_BATCH"], num_dom,
                            seed=h["SEED"], split="val", pl_mode="hide", pl_domain_id=h["PL_DOMAIN_ID"])

    test_ds  = make_dataset(X_test, d_test, l_test.squeeze(), w_test, h["AE_BATCH"], num_dom,
                            seed=h["SEED"], split="test", pl_mode="use", pl_domain_id=h["PL_DOMAIN_ID"])
    if DEBUG:

        print("Important facts train :\n")
        for batch in train_ds:
            sig, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los) = batch
            domains = tf.argmax(y_dom, axis=-1).numpy()

            print("domain values:", *domains,sep="    ")
            print("sw_rec values:", *sw_rec.numpy(),sep="  ")
            print("sw_dom values:", *sw_dom.numpy(),sep="  ")
            

            print("sw_los values:", *sw_los.numpy(),sep="  ")  # convert to numpy if needed
            print("y_los values:", *y_los.numpy().reshape(-1),sep="  ")  # convert to numpy if needed
            break
        print("Important facts validation :\n")
        for batch in val_ds:
            sig, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los) = batch
            domains = tf.argmax(y_dom, axis=-1).numpy()

            print("domain values:", *domains,sep="    ")
            print("sw_rec values:", *sw_rec.numpy(),sep="  ")
            print("sw_dom values:", *sw_dom.numpy(),sep="  ")
            print("sw_los values:", *sw_los.numpy(),sep="  ")  # convert to numpy if needed
            print("y_los values:", *y_los.numpy().reshape(-1),sep="  ")  # convert to numpy if needed
            break 

    #selecting the device to run the algorithm 
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    #BFC_mean = BinaryFocalCrossentropy(gamma=h["FOCAL_GAMMA"], alpha=h["FOCAL_ALPHA"],from_logits=False, reduction='sum_over_batch_size')  
    BFC_mean = BinaryFocalCrossentropy(gamma=2.0, alpha=0.25,from_logits=False, reduction='sum_over_batch_size')  

    def reconstruction_loss(y_true, y_pred):
        raw = time_freq_log_loss(y_true, y_pred)        # (batch,) or (batch,features)
      
        if DEBUG:
            print("the raw elements: ",*raw.numpy())
            print("the sum elements: ",np.sum(raw.numpy()))
            print("the mean elements: ",np.mean(raw.numpy()))
            print("counts of element: ",len(raw.numpy()))
            print("the mean(tensorflow) elements: ",tf.reduce_mean(raw).numpy())
        return tf.reduce_mean(raw)

    def domain_loss(y_true, y_pred, sw):
        ce_dom = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=h["DOM_LABEL_SMOOTH"], reduction='none')
        per_sample = ce_dom(y_true, y_pred)             # (batch,)
        if DEBUG:
            print("y_true : ",*y_true.numpy().reshape(-1))
            print("y_pred : ",*y_pred.numpy().reshape(-1))
            print("sw_numpy: ",sw.numpy())
            print("per_sample: ",*per_sample.numpy())
            print("mean per_sample: ",np.mean(per_sample.numpy()))
            print("sum _weighted_mean: ",tf.reduce_mean(per_sample))
            print("mean per_sample with weight: ",np.sum(per_sample*sw.numpy())/np.sum(sw.numpy())+ 1e-8)
            print("_weighted_mean: ",_weighted_mean(per_sample, sw))
            
        return _weighted_mean(per_sample, sw)

    def los_loss(y_true, y_pred, sw):
        # 1) build mask of “valid” entries (where sw != 0)
        sw     = tf.reshape(tf.cast(sw,     tf.float32), (-1,))
        mask = tf.not_equal(sw, 0)

        # 2) flatten out only those valid entries
        y_flat    = tf.boolean_mask(y_true, mask)
        pred_flat = tf.boolean_mask(y_pred, mask)

        return BFC_mean(y_flat, pred_flat)  
        #return focal_tversky(y_flat, pred_flat)

    if DEBUG:
        print(f"Using device: {device}")


    with tf.device(device):
        # ---- build model ----

       
        ae, grl = build_DA_AE(
            seq_len=X.shape[1], n_domains=num_dom, latent_dim=h["LATENT_DIM"],
            ENC_CONST_COEF=h["ENC_CONST_COEF"], DEC_CONST_COEF=h["DEC_CONST_COEF"]
        )
        #taking the los_logits head before the sigmoid head
        los_logits_head = tf.keras.Model(
        inputs=ae.input,
        outputs=ae.get_layer("los_logit").output,
        name="los_logits_head"
    )
        
        
              

        best_saver = BestAccSaver(ae, h["save_plots"], best_name="best_val_weights.weights.h5", last_name="last_epoch_weight.weights.h5")
        prog_unfreeze = ProgressiveUnfreeze(ae, step_epochs=h["UNFREEZE_STEP"])
        early_stop = SimpleEarlyStopping(monitor="val_los_acc", patience=h["AE_PATIENCE"])
        early_stop.set_model(ae)


        # ---- optimizer & sched ----
        steps_per_epoch = math.ceil(len(X_tr) / h["AE_BATCH"])
        total_steps     = max(1, h["AE_EPOCHS"] * steps_per_epoch)
        warmup_steps    = h["LR_WARMUP_EPOCHS"] * steps_per_epoch  # int

        lr_sched = WarmupThenCosine(
            base_lr=h["BASE_LR"],
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            alpha=h["COSINE_ALPHA"],
        )
     

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=h["CLIPNORM"])


        # ---- losses (per-sample → weighted mean) ----
        
  
                # ---- dynamic loss-weights (tensor we pass to @tf.function) ----
        lw_start = tf.constant([h["LW_REC_START"], h["LW_DOM_START"], h["LW_LOS_START"]], tf.float32)
        lw_end   = tf.constant([h["LW_REC_END"],   h["LW_DOM_END"],   h["LW_LOS_END"]],   tf.float32)


        def interp_lw(epoch, total):
            p = tf.cast(epoch, tf.float32) / tf.cast(tf.maximum(1, total-1), tf.float32)
            return (1.0 - p) * lw_start + p * lw_end
        
        
        
        


        # ---- metrics ----
        train_dom_acc = tf.keras.metrics.CategoricalAccuracy(name="train_dom_acc")
        train_dom_loss = tf.keras.metrics.Mean(name="train_dom_loss")
        train_los_acc = tf.keras.metrics.BinaryAccuracy(name="train_los_acc", threshold=h["METRIC_THRESHOLD"])
        train_los_loss = tf.keras.metrics.Mean(name="train_los_loss")
        train_recon_loss = tf.keras.metrics.Mean(name="train_recon_loss")
        train_cons_loss   = tf.keras.metrics.Mean(name="train_cons_loss")
        train_cons_wmean  = tf.keras.metrics.Mean(name="train_cons_wmean")  # avg weight used for KL
        ###################################################################################
        val_dom_acc   = tf.keras.metrics.CategoricalAccuracy(name="val_dom_acc")
        val_dom_loss  = tf.keras.metrics.Mean(name="val_dom_loss")
        val_los_acc = tf.keras.metrics.BinaryAccuracy(name="val_los_acc", threshold=h["METRIC_THRESHOLD"])
        val_los_loss  = tf.keras.metrics.Mean(name="val_los_loss")
        val_recon_loss = tf.keras.metrics.Mean(name="val_recon_loss")
        val_cons_loss     = tf.keras.metrics.Mean(name="val_cons_loss")
        val_cons_wmean    = tf.keras.metrics.Mean(name="val_cons_wmean")
        #################################################################################
        test_dom_acc  = tf.keras.metrics.CategoricalAccuracy(name="test_dom_acc")
        test_los_acc  = tf.keras.metrics.BinaryAccuracy(name="test_los_acc", threshold=h["METRIC_THRESHOLD"])
        test_los_loss = tf.keras.metrics.Mean(name="test_los_loss")
        test_dom_loss = tf.keras.metrics.Mean(name="test_dom_loss")
        test_recon_loss = tf.keras.metrics.Mean(name="test_recon_loss")
        lw_test = tf.constant([1.0, 1.0, 1.0], tf.float32)


        @tf.function  # turn on jit_compile=True later only if it shows a real speedup
        def train_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec, lam_cons):
            with tf.GradientTape() as tape:
                # Forward
                pred_rec, pred_dom, pred_los = ae(x, training=True)

                # ---- dtypes ----
                y_rec   = tf.cast(y_rec,  tf.float32)
                y_dom   = tf.cast(y_dom,  tf.float32)
                y_los   = tf.cast(y_los,  tf.float32)
                sw_rec  = tf.cast(sw_rec, tf.float32)
                sw_dom  = tf.cast(sw_dom, tf.float32)
                sw_los  = tf.cast(sw_los, tf.float32)
                pred_rec = tf.cast(pred_rec, tf.float32)
                pred_dom = tf.cast(pred_dom, tf.float32)
                pred_los = tf.cast(pred_los, tf.float32)

                # ---- losses ----
                Lr = reconstruction_loss(y_rec, pred_rec)

                pred_los = tf.clip_by_value(pred_los, 1e-6, 1.0 - 1e-6)
                Ll = los_loss(y_los, pred_los, sw_los)

                # ---- CDAN-E: entropy weight (stop grad) ----
                g2 = tf.concat([pred_los, 1.0 - pred_los], axis=-1)  # (B,2)
                H  = -tf.reduce_sum(g2 * tf.math.log(g2 + 1e-6), axis=-1) / tf.math.log(tf.constant(2.0, tf.float32))
                if DEBUG:
                    mask_H=tf.where(tf.math.is_inf(H))
                    print("count of INF H: ",len(mask_H))
                H  = tf.where(tf.math.is_finite(H), H, 1.0)
                beta = tf.cast(h.get("ENTROPY_BETA", 2.0), tf.float32)
                w_e  = tf.stop_gradient(tf.exp(-beta * H))           # (B,)

                # ---- domain reweight (no scatter; fully GPU-friendly) ----
                dom_w = tf.ones([num_dom], tf.float32)
                pl_domain_id = tf.cast(h["PL_DOMAIN_ID"], tf.int64)              # keep indices int64
                target_val   = tf.cast(h["DOM_WEIGHT_TARGET"], tf.float32)
                oh = tf.one_hot(pl_domain_id, depth=num_dom, dtype=tf.float32)   # (num_dom,)
                dom_w = dom_w + (target_val - 1.0) * oh                          # set only target domain to target_val

                # per-sample domain weight
                per_dom_w = tf.reduce_sum(y_dom * dom_w[tf.newaxis, :], axis=-1)  # (B,)
                sw_dom_vec = tf.reshape(sw_dom, (-1,))                             # (B,)
                sw_dom_eff = sw_dom_vec * per_dom_w * w_e                          # (B,)

                if DEBUG:
                    print("Effective domain weights (first 20):", *sw_dom_eff.numpy()[:20])
                    print("g2: ",g2.numpy())
                    print("H: ",H.numpy())
                    print("w_e: ",w_e.numpy())
                    print("beta: ",beta)
                    print("dom_w: ",dom_w.numpy())
                    print("per_dom_w: ",per_dom_w.numpy())
                    print("sw_dom_vec: ",sw_dom_vec.numpy())
                    
                    print("pl_domain_id: ",pl_domain_id.numpy())
                    print("target_val: ",target_val.numpy())
                    print("y_dom: ",y_dom.numpy())
                    print("num_dom: ",num_dom)
                    print("sw_dom_eff: ",sw_dom_eff.numpy())

                Ld = domain_loss(y_dom, pred_dom, sw_dom_eff)

                total = lw_vec[0] * Lr + lw_vec[1] * Ld + lw_vec[2] * Ll

                # ---- consistency regularization ----
                Lcons = tf.constant(0.0, tf.float32)
                cons_w_mean = tf.constant(0.0, tf.float32)

                if h["CONS_ENABLE"]:
                    dom_ids = tf.argmax(y_dom, axis=-1, output_type=tf.int64)     # int64
                    mask_target = tf.equal(dom_ids, pl_domain_id)                  # (B,)

                    if h.get("CONS_ON", "target_all") == "target_unlabeled":
                        unlabeled = tf.equal(tf.reshape(sw_los, (-1,)), tf.constant(0.0, tf.float32))
                        mask_target = mask_target & unlabeled

                    if tf.reduce_any(mask_target):
                        x_s = strong_aug(x, h["AUG_NOISE_STD"], h["AUG_GAIN_JITTER"])

                        # Teacher logits (stable)
                        p_teacher = tf.stop_gradient(pred_los)                     # (B,1)
                        logits_w  = tf.math.log(p_teacher + 1e-6) - tf.math.log(1.0 - p_teacher + 1e-6)

                        # Student logits
                        logits_s = tf.cast(los_logits_head(x_s, training=True), tf.float32)

                        # confidence weights
                        conf  = tf.maximum(p_teacher, 1.0 - p_teacher)             # (B,1)
                        w_cons = tf.pow(conf, tf.cast(h["CONS_CONF_GAMMA"], tf.float32))
                        mask_f = tf.cast(mask_target, tf.float32)[:, None]
                        w_cons = w_cons * mask_f                                   # (B,1)

                        cons_w_mean = tf.reduce_sum(w_cons) / (tf.reduce_sum(mask_f) + 1e-8)

                        Lcons = kl_consistency_from_logits(
                            student_logits=logits_s,
                            teacher_logits=tf.cast(logits_w, tf.float32),
                            T=h["CONS_T"],
                            weight=tf.reshape(w_cons, (-1,))
                        )
                        total += tf.cast(lam_cons, tf.float32) * Lcons

            # ---- single fused clip (faster) ----
            grads = tape.gradient(total, ae.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, ae.trainable_variables))

            # ---- metrics (no prints; avoid host syncs) ----
            train_dom_acc.update_state(y_dom, pred_dom)
            train_los_acc.update_state(y_los, pred_los, sample_weight=sw_los)
            train_dom_loss.update_state(Ld)
            train_los_loss.update_state(Ll)
            train_recon_loss.update_state(Lr)
            if h["CONS_ENABLE"]:
                train_cons_loss.update_state(Lcons)
                train_cons_wmean.update_state(cons_w_mean)

            return total
        






        @tf.function# (jit_compile=False)
        def val_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec, lam_cons):
            # Forward (eval mode)
            pred_rec, pred_dom, pred_los = ae(x, training=False)
            pred_los = tf.clip_by_value(pred_los, 1e-6, 1.0 - 1e-6)
            pred_los = tf.where(tf.math.is_finite(pred_los), pred_los, 0.0)

            # Dtypes
            y_rec = tf.cast(y_rec, tf.float32)
            y_dom = tf.cast(y_dom, tf.float32)
            y_los = tf.cast(y_los, tf.float32)
            sw_rec = tf.cast(sw_rec, tf.float32)
            sw_dom = tf.cast(sw_dom, tf.float32)
            sw_los = tf.cast(sw_los, tf.float32)
            pred_rec = tf.cast(pred_rec, tf.float32)
            pred_dom = tf.cast(pred_dom, tf.float32)
            pred_los = tf.cast(pred_los, tf.float32)

            # ----- Base losses -----
            Lr = reconstruction_loss(y_rec, pred_rec)
            Ll = los_loss(y_los, pred_los, sw_los)

            # ----- CDAN-E: entropy-weighted domain loss -----
            # Ensure probabilities are strictly in (0,1)
            pred_los = tf.clip_by_value(pred_los, 1e-6, 1.0 - 1e-6)
            pred_los = tf.where(tf.math.is_finite(pred_los), pred_los, 0.5)

            g2 = tf.concat([pred_los, 1.0 - pred_los], axis=-1)          # (B,2)
            H  = -tf.reduce_sum(g2 * tf.math.log(g2 + 1e-6), axis=-1) / tf.math.log(2.0)
            H  = tf.where(tf.math.is_finite(H), H, 1.0)                  # worst case
            beta = tf.cast(h.get("ENTROPY_BETA", 2.0), tf.float32)
            w_e  = tf.exp(-beta * H)
            w_e  = tf.stop_gradient(w_e)
            # optional bound: w_e = tf.clip_by_value(w_e, 0.5, 3.0)

            sw_dom_vec = tf.reshape(sw_dom, (-1,))                       # (B,)
            sw_dom_eff = sw_dom_vec * w_e                                # (B,)
            Ld = domain_loss(y_dom, pred_dom, sw_dom_eff)

            total = lw_vec[0]*Lr + lw_vec[1]*Ld + lw_vec[2]*Ll

            # ----- Consistency regularization (eval/monitoring) -----
            if h["CONS_ENABLE"]:
                dom_ids = tf.argmax(y_dom, axis=-1, output_type=tf.int32)
                mask_target = tf.equal(dom_ids, tf.cast(h["PL_DOMAIN_ID"], tf.int32))

                if tf.reduce_any(mask_target):
                    # Strong aug for student
                    x_s = strong_aug(x, h["AUG_NOISE_STD"], h["AUG_GAIN_JITTER"])
                    logits_s = tf.cast(los_logits_head(x_s, training=False), tf.float32)

                    # Teacher logits derived from existing pred_los (saves a forward pass)
                    # logits_w = logit(p) = log(p) - log(1-p)
                    logits_w = tf.math.log(pred_los) - tf.math.log(1.0 - pred_los)

                    # Confidence weights
                    p_w = pred_los  # already clipped
                    conf = tf.maximum(p_w, 1.0 - p_w)                                  # (B,1)
                    w_cons = tf.pow(conf, tf.cast(h["CONS_CONF_GAMMA"], tf.float32))   # (B,1)
                    w_cons = w_cons * tf.cast(mask_target[:, None], tf.float32)        # mask target only

                    Lcons_val = kl_consistency_from_logits(
                        student_logits=logits_s,
                        teacher_logits=tf.cast(logits_w, tf.float32),
                        T=h["CONS_T"],
                        weight=tf.reshape(w_cons, (-1,))
                    )

                    # Log the consistency loss and its effective weight mean
                    val_cons_loss.update_state(Lcons_val)
                    denom = tf.reduce_sum(tf.cast(mask_target, tf.float32)) + 1e-8
                    val_cons_wmean.update_state(tf.reduce_sum(w_cons) / denom)

                    # Optionally include in the displayed validation total
                    
                    
                    total += lam_cons * Lcons_val

            # Metrics
            val_dom_acc.update_state(y_dom, pred_dom)
            val_los_acc.update_state(y_los, pred_los, sample_weight=sw_los)
            val_dom_loss.update_state(Ld)
            val_los_loss.update_state(Ll)
            val_recon_loss.update_state(Lr)

          

            return total







        @tf.function#(jit_compile=True)
        def test_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec):
            # -------- forward (eval mode) --------
            pred_rec, pred_dom, pred_los = ae(x, training=False)

            # dtypes
            y_rec = tf.cast(y_rec, tf.float32)
            y_dom = tf.cast(y_dom, tf.float32)
            y_los = tf.cast(y_los, tf.float32)
            sw_rec = tf.cast(sw_rec, tf.float32)
            sw_dom = tf.cast(sw_dom, tf.float32)
            sw_los = tf.cast(sw_los, tf.float32)
            pred_rec = tf.cast(pred_rec, tf.float32)
            pred_dom = tf.cast(pred_dom, tf.float32)
            pred_los = tf.cast(pred_los, tf.float32)

            # clip probs before any log/entropy math
            pred_los = tf.clip_by_value(pred_los, 1e-6, 1.0 - 1e-6)
            pred_los = tf.where(tf.math.is_finite(pred_los), pred_los, 0.5)

            # -------- base losses --------
            Lr = reconstruction_loss(y_rec, pred_rec)
            Ll = los_loss(y_los, pred_los, sw_los)

            # -------- CDAN-E: entropy weight for domain loss (eval-only) --------
            g2 = tf.concat([pred_los, 1.0 - pred_los], axis=-1)                # (B,2)
            H  = -tf.reduce_sum(g2 * tf.math.log(g2 + 1e-6), axis=-1) / tf.math.log(2.0)
            H  = tf.where(tf.math.is_finite(H), H, 1.0)                        # fallback
            beta = tf.cast(h.get("ENTROPY_BETA", 2.0), tf.float32)
            w_e  = tf.exp(-beta * H)
            w_e  = tf.stop_gradient(w_e)
            # optional: w_e = tf.clip_by_value(w_e, 0.5, 3.0)

            sw_dom_vec = tf.reshape(sw_dom, (-1,))
            sw_dom_eff = sw_dom_vec * w_e
            Ld = domain_loss(y_dom, pred_dom, sw_dom_eff)

            total = lw_vec[0] * Lr + lw_vec[1] * Ld + lw_vec[2] * Ll

            # -------- optional: LOG consistency on test (do NOT add to total) --------
            # Enable with: h["CONS_ENABLE"]=True and h["CONS_LOG_IN_TEST"]=True
            if h.get("CONS_ENABLE", False) and h.get("CONS_LOG_IN_TEST", False):
                dom_ids = tf.argmax(y_dom, axis=-1, output_type=tf.int32)
                mask_target = tf.equal(dom_ids, tf.cast(h["PL_DOMAIN_ID"], tf.int32))
                if h.get("CONS_ON", "target_all") == "target_unlabeled":
                    mask_target = mask_target & tf.equal(tf.reshape(sw_los, (-1,)), 0.0)

                if tf.reduce_any(mask_target):
                    # student on strong aug, still eval mode in test
                    x_s = strong_aug(x, h["AUG_NOISE_STD"], h["AUG_GAIN_JITTER"])
                    logits_s = tf.cast(los_logits_head(x_s, training=False), tf.float32)

                    # teacher logits from existing probs (no extra forward, no tape)
                    p_teacher = tf.stop_gradient(pred_los)                      # (B,1)
                    logits_w  = tf.math.log(p_teacher) - tf.math.log(1.0 - p_teacher)

                    # confidence weights
                    conf  = tf.maximum(p_teacher, 1.0 - p_teacher)              # (B,1)
                    w_cons = tf.pow(conf, tf.cast(h["CONS_CONF_GAMMA"], tf.float32))
                    mask_f = tf.cast(mask_target, tf.float32)[:, None]
                    w_cons = w_cons * mask_f

                 

            # -------- metrics --------
            test_dom_acc.update_state(y_dom, pred_dom)
            test_los_acc.update_state(y_los, pred_los, sample_weight=sw_los)
            test_dom_loss.update_state(Ld)
            test_los_loss.update_state(Ll)
            test_recon_loss.update_state(Lr)

           

            return total

        
        
        
       

        grl_cb = GRLSchedule(grl, h["AE_EPOCHS"],
                lambda_min=0.0, lambda_max=h["GRL_LAMBDA_MAX"], peak_frac=h["GRL_PEAK_FRAC"]
            )

       
        history = {
            "loss":[], "train_dom_acc":[], "train_dom_loss":[], "train_los_acc":[], "train_los_loss":[], "train_recon_loss":[],
            "val_loss":[], "val_dom_acc":[], "val_dom_loss":[], "val_los_acc":[], "val_los_loss":[], "val_recon_loss":[],
            "test_loss":[], "test_dom_acc":[], "test_dom_loss":[], "test_los_acc":[], "test_los_loss":[], "test_recon_loss":[],
            "grl_lambda":[],
            # NEW:
            "train_cons_loss":[], "val_cons_loss":[],
            "train_cons_wmean":[], "val_cons_wmean":[],
            "cons_lambda":[]
        }

            

        start = time.time()
        for epoch in range(h["AE_EPOCHS"]):
            grl_cb.on_epoch_begin(epoch)
            prog_unfreeze.on_epoch_begin(epoch)
            # at epoch start
            train_cons_loss.reset_state()
            train_cons_wmean.reset_state()
            val_cons_loss.reset_state()
            val_cons_wmean.reset_state()

           
            




            # linearly interpolate loss weights this epoch
            lw_vec = interp_lw(epoch, h["AE_EPOCHS"])
          #  print("reconstruction loss weight:", lw_vec[0].numpy(),"domain loss weight:", lw_vec[1].numpy(),"latent loss weight:", lw_vec[2].numpy())

            # --- train
            train_dom_acc.reset_state()
            train_los_acc.reset_state()
            train_dom_loss.reset_state()
            train_los_loss.reset_state()
            train_recon_loss.reset_state()

            train_loss = tf.keras.metrics.Mean()
           # batch_num=0
            #temp_total=0.0

            lam_cons = cons_lambda(epoch,h)  
            batch_num = 0

            for x, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los) in train_ds:
                #batch_num += 1
               # print("batch_num:", batch_num)

                
                total = train_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec, lam_cons)
                train_loss.update_state(total)

            # --- val
            val_dom_acc.reset_state()
            val_dom_loss.reset_state()
            val_los_acc.reset_state()
            val_los_loss.reset_state()
            val_recon_loss.reset_state()
            val_loss = tf.keras.metrics.Mean()
            
            for x, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los) in val_ds:
                 total = val_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec, lam_cons)
                 val_loss.update_state(total)


            #$#$#$#$#$
            test_dom_acc.reset_state()
            test_los_acc.reset_state()
            test_dom_loss.reset_state()
            test_los_loss.reset_state()
            test_recon_loss.reset_state()
            #$#$#$#$#$
            test_loss = tf.keras.metrics.Mean()
            for x, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los) in test_ds:
                # use lw_vec or lw_test (see note above)
                total = test_step(x, y_rec, y_dom, y_los, sw_rec, sw_dom, sw_los, lw_vec)
                test_loss.update_state(total)
           


            # #$$##$#$#$#$    

            # log
            # inside each epoch after evaluation:
            history["loss"].append(float(train_loss.result()))
            history["train_dom_acc"].append(float(train_dom_acc.result()))
            history["train_dom_loss"].append(float(train_dom_loss.result()))
            history["train_los_acc"].append(float(train_los_acc.result()))
            history["train_los_loss"].append(float(train_los_loss.result()))
            history["train_recon_loss"].append(float(train_recon_loss.result()))

            history["val_loss"].append(float(val_loss.result()))
            history["val_dom_acc"].append(float(val_dom_acc.result()))
            history["val_dom_loss"].append(float(val_dom_loss.result()))
            history["val_los_acc"].append(float(val_los_acc.result()))
            history["val_los_loss"].append(float(val_los_loss.result()))
            history["val_recon_loss"].append(float(val_recon_loss.result()))

            history["test_loss"].append(float(test_loss.result()))
            history["test_dom_acc"].append(float(test_dom_acc.result()))
            history["test_dom_loss"].append(float(test_dom_loss.result()))
            history["test_los_acc"].append(float(test_los_acc.result()))
            history["test_los_loss"].append(float(test_los_loss.result()))
            history["test_recon_loss"].append(float(test_recon_loss.result()))
            history["grl_lambda"].append(float(grl.hp_lambda.numpy()))

            history["train_cons_loss"].append(float(train_cons_loss.result()))
            history["train_cons_wmean"].append(float(train_cons_wmean.result()))
            history["val_cons_loss"].append(float(val_cons_loss.result()))
            history["val_cons_wmean"].append(float(val_cons_wmean.result()))
            history["cons_lambda"].append(float(lam_cons.numpy()))

            val_acc = history["val_los_acc"][-1]

        # 2) update best / wait
        
                    
            


            val_acc_epoch = float(val_los_acc.result())
            best_saver.on_epoch_end(epoch, val_acc_epoch)

           

            print(
                f"Epoch {epoch+1:03d}/{h['AE_EPOCHS']} "
                f"loss={history['loss'][-1]:.4f} "
                f"dom_acc={history['train_dom_acc'][-1]:.3f} "
                f"los_acc={history['train_los_acc'][-1]:.3f} | "
                f"val_loss={history['val_loss'][-1]:.4f} "
                f"val_dom_acc={history['val_dom_acc'][-1]:.3f} "
                f"val_los_acc={history['val_los_acc'][-1]:.3f} | "
                f"test_loss={history['test_loss'][-1]:.4f} "
                f"test_dom_acc={history['test_dom_acc'][-1]:.3f} "
                f"test_los_acc={history['test_los_acc'][-1]:.3f}"
            )
      
            early_stop.on_epoch_end(epoch, {"val_los_acc": val_acc})
            if getattr(early_stop.model, "stop_training", False):
                break
                    

        # restore best weights if any, and persist
      
        best_saver.on_train_end()
        

 
    def model_prediction_with_weights(model, weight_name, Weights=None):
        model.load_weights(os.path.join(h["save_plots"], weight_name))
        los_model = tf.keras.Model(
                inputs  = model.input,
                outputs = model.get_layer("los_prob").output,
                name    = "los_only_model"

            )
        los_model.compile(loss=BFC_mean , metrics=["accuracy",tn])
        return los_model
    
    BFC_none = BinaryFocalCrossentropy(
    gamma=h["FOCAL_GAMMA"], alpha=h["FOCAL_ALPHA"],
    from_logits=False, reduction='none')

    def bernoulli_entropy_from_logits(logits, base='e'):
        """Mean Bernoulli entropy from *logits* (scalar per example)."""
        x = tf.convert_to_tensor(logits, tf.float32)
        p = tf.nn.sigmoid(x)
        eps = tf.constant(1e-8, tf.float32)
        H = -(p * tf.math.log(p + eps) + (1.0 - p) * tf.math.log(1.0 - p + eps))
        Hm = tf.reduce_mean(H)
        return Hm / tf.math.log(tf.constant(2.0, tf.float32)) if base == 2 else Hm

    def entropy_on_adaption_logits(los_logits_model, X2d, base='e'):
        logits = los_logits_model.predict(X2d[..., None], verbose=0).reshape(-1)
        return float(bernoulli_entropy_from_logits(logits, base=base).numpy())

    
          
    def model_for_entropy(model, weight_dir, weight_name):
        model.load_weights(os.path.join(weight_dir, weight_name))
        logits = model.get_layer("los_logit").output
        return tf.keras.Model(inputs=model.input, outputs=logits, name="los_logits_model")
    # 1) Evaluate last-epoch weights on TEST
    los_model = model_prediction_with_weights(ae, "last_epoch_weight.weights.h5", Weights)
    TEST_res_last_epoch = predict_los_only(CIRS, LosLabels, "TEST", los_model, WEIGHTS=Weights)

    # 2) Load BEST weights (for all metrics below)
    los_model = model_prediction_with_weights(ae, "best_val_weights.weights.h5", Weights)   # prob head
    los_model_entropy = model_for_entropy(ae, h["save_plots"], "best_val_weights.weights.h5")  # logits head

    # -------- gather TRAIN arrays --------
    Xtr, ltr, wtr = [], [], []
    for x, (_, _, y_los), (_, _, sw_los) in train_ds:
        Xtr.append(x.numpy())
        ltr.append(y_los.numpy())
        wtr.append(sw_los.numpy())

    Xtr = np.concatenate(Xtr, axis=0)
    ltr = np.concatenate(ltr, axis=0).squeeze()
    wtr = np.concatenate(wtr, axis=0).squeeze()

    # -------- predictions (use PROB model; threshold from config) --------
    
    los_probs = los_model.predict(Xtr, verbose=0).reshape(-1)  # probabilities in (0,1)

    mask = (wtr != 0)                     # supervised samples
    ltr_filtered = ltr[mask]
    probs_filt   = los_probs[mask]
    preds_filt   = (probs_filt >= float(h["METRIC_THRESHOLD"])).astype(int)

    # -------- supervised TRAIN metrics (correctly use probs where required) --------
    accuracy_train = accuracy_score(ltr_filtered, preds_filt)
    f1_train       = f1_score(ltr_filtered, preds_filt)
    cm_train       = confusion_matrix(ltr_filtered, preds_filt)

    # ---- entropy on UNLABELED (w==0) using LOGITS model (correct entropy input) ----
    train_logits = los_model_entropy.predict(Xtr, verbose=0).reshape(-1)   # logits
    unlabeled_mask = ~mask
    # NOTE: use your new helper bernoulli_entropy_from_logits() (see my previous message)
    train_entropy = float(bernoulli_entropy_from_logits(train_logits[unlabeled_mask]).numpy())
    train_mean_entropy= mean_entropy(train_logits[unlabeled_mask])
    f1_entropy_train = f1_train - train_entropy

    # also report Bernoulli entropy from PROBS for reference (different definition)
    train_binary_entropy = binary_entropy(los_probs[unlabeled_mask])  # expects probs

    # ---- BCE/Dice (BCE must use probabilities, not hard labels) ----
    # If you have a custom dice, use it; otherwise ensure this symbol exists in your codebase.
    train_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.cast(ltr_filtered, tf.float32),
        tf.cast(probs_filt,   tf.float32)
    ))
    # If you rely on a custom dice: dice_coef(y_true, y_pred_binary)
    train_dsc = tf.keras.losses.dice(  # <-- if this is not your custom, replace with your own dice function
        tf.cast(ltr_filtered, tf.float32),
        tf.cast(preds_filt,   tf.float32)
    )
    train_dice_loss = 1.0 - train_dsc
    train_bce_dice  = train_bce + train_dice_loss

    # other per-sample stats (your custom helpers)
    train_tp = tp(ltr_filtered, preds_filt)
    train_tn = tn(ltr_filtered, preds_filt)
    train_balanced_accuracy_score=balanced_accuracy_score(ltr_filtered, preds_filt)
    train_focal_tversky = focal_tversky(ltr_filtered, preds_filt)
    train_weighted_bce_logits = weighted_bce_logits(
        tf.cast(ltr_filtered, tf.float32),
        tf.cast(probs_filt, tf.float32)
    )

    # focal cross-entropy: use probabilities, read alpha/gamma from h if present
    train_binary_focal_crossentropy = tf.reduce_mean(
        tf.keras.losses.binary_focal_crossentropy(
            tf.cast(ltr_filtered, tf.float32),
            tf.cast(probs_filt,   tf.float32),
            apply_class_balancing=False,
            alpha=h.get("FOCAL_ALPHA", 0.25),
            gamma=h.get("FOCAL_GAMMA", 2.0),
            from_logits=False,
            label_smoothing=0.0,
            axis=-1
        )
    )
    if DEBUG:
        print(f"Training accuracy (wv≠0): {accuracy_train:.4f}")
        print(f"Training   F1    (wv≠0): {f1_train:.4f}")
        print(f"Training F1 (entropy) (wv≠0): {f1_entropy_train:.4f}")
        print("Training confusion matrix (wv≠0):")
        print(cm_train)


    Xv, lv, wv = [], [], []
    for x, (_, _, y_los), (_, _, sw_los) in val_ds:
        Xv.append(x.numpy())
        lv.append(y_los.numpy())
        wv.append(sw_los.numpy())

    Xv = np.concatenate(Xv, axis=0)
    lv = np.concatenate(lv, axis=0).squeeze()
    wv = np.concatenate(wv, axis=0).squeeze()

    # compute predictions
    los_probs = ae.predict(Xv, verbose=0)[2].squeeze()
    preds_bin = (los_probs >= float(h["METRIC_THRESHOLD"])).astype(int)

    # mask out the wv == 0 samples in one line each
    mask = (wv != 0)
    lv_filtered    = lv[mask]
    probs_filt     = los_probs[mask]
    preds_bin = (los_probs >= float(h["METRIC_THRESHOLD"])).astype(int)
    preds_filtered = preds_bin[mask]
    val_auroc = float(roc_auc_score(lv_filtered,probs_filt))

    # compute metrics
    accuracy_valid = accuracy_score(lv_filtered, preds_filtered)
    f1_valid       = f1_score    (lv_filtered, preds_filtered)
    cm_valid       = confusion_matrix(lv_filtered, preds_filtered)
    
    val_logits =los_model_entropy.predict(Xv, verbose=0).reshape(-1)

    valid_entropy= mean_entropy(val_logits[~mask])
    validation_binary_entropy = binary_entropy(los_probs[~mask])
    f1_entropy_valid = f1_valid - valid_entropy
    validation_dsc = tf.keras.losses.dice(
    tf.cast(lv_filtered, tf.float32),
    tf.cast(probs_filt, tf.float32)
)
    
    

    validation_dice_loss = 1 - validation_dsc
    validation_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
    tf.cast(lv_filtered, tf.float32),
    tf.cast(probs_filt, tf.float32)
))
    validation_bce_dice = validation_bce + validation_dice_loss

    validation_tp = tp(lv_filtered, preds_filtered)
    validation_tn = tn(lv_filtered, preds_filtered)
    
    validation_balanced_accuracy_score=balanced_accuracy_score(lv_filtered, preds_filtered)
    validation_focal_tversky = focal_tversky(lv_filtered, preds_filtered)
    validation_binary_focal_crossentropy = tf.reduce_mean(
        tf.keras.losses.binary_focal_crossentropy(
            tf.cast(lv_filtered, tf.float32),
            tf.cast(probs_filt, tf.float32),
            apply_class_balancing=False,
            alpha=h.get("FOCAL_ALPHA", 0.25),
            gamma=h.get("FOCAL_GAMMA", 2.0),
            from_logits=False,
            label_smoothing=0.0,
            axis=-1
        )
    )
    validation_weighted_bce_logits = weighted_bce_logits(
    tf.cast(lv_filtered, tf.float32),
    tf.cast(probs_filt, tf.float32)
)
    
   


    if DEBUG:
        print(f"Validation accuracy (wv≠0): {accuracy_valid:.4f}")
        print(f"Validation   F1    (wv≠0): {f1_valid:.4f}")
        print(f"Validation F1 (entropy) (wv≠0): {f1_entropy_valid:.4f}")
        print("Validation confusion matrix (wv≠0):")
        print(cm_valid)

        # ---------------- timing & history ----------------
    training_time = time.time() - start

    # You don't really need a wrapper class; return the dict directly.
    # If other code expects ".history", use a tiny namespace instead:
    from types import SimpleNamespace
    ae_history = SimpleNamespace(history=history)

    plot_all_histories(history, h["save_plots"])
    plot_history(history, h["save_plots"])


    # ---------------- split-wise summaries using BEST weights ----------------
    TRAIN1_res  = predict_los_only(CIRS, LosLabels, "TRAIN1",  los_model, WEIGHTS=Weights)
    TRAIN2_res  = predict_los_only(CIRS, LosLabels, "TRAIN2",  los_model, WEIGHTS=Weights)
    ADAPTION_res= predict_los_only(CIRS, LosLabels, "ADAPTION",los_model, WEIGHTS=Weights)
    TEST_res    = predict_los_only(CIRS, LosLabels, "TEST",    los_model, WEIGHTS=Weights)

    # ======================================================================
    # Helpers (DRY, precise semantics: probs vs logits)
    # ======================================================================
    def _ensure_3d(X):
        return X[..., None] if X.ndim == 2 else X

    def predict_probs(model_prob_head, X2d, weights=None):
        """Predict P(class=1). If weights given, filter where w!=0."""
        X3d = _ensure_3d(X2d)
        if weights is not None:
            X3d = X3d[weights != 0]
        return model_prob_head.predict(X3d, verbose=0).reshape(-1)

    def predict_logits(model_logit_head, X2d, weights=None):
        """Predict logits for class=1. If weights given, filter where w!=0."""
        X3d = _ensure_3d(X2d)
        if weights is not None:
            X3d = X3d[weights != 0]
        return model_logit_head.predict(X3d, verbose=0).reshape(-1)

    def f1_on_train12(CIRS, Labels, model_prob_head, threshold=None, optimize_threshold=False, weights=None):
        X_tr12 = np.concatenate([CIRS["TRAIN1"], CIRS["TRAIN2"]], axis=0)
        y_tr12 = np.concatenate([np.asarray(Labels["TRAIN1"]), np.asarray(Labels["TRAIN2"])]).astype(int)

        if weights is not None:
            w_tr12 = np.concatenate([np.asarray(weights["TRAIN1"]), np.asarray(weights["TRAIN2"])]).astype(float)
            mask = (w_tr12 != 0)
            X_tr12 = X_tr12[mask]
            y_tr12 = y_tr12[mask]

        probs = predict_probs(model_prob_head, X_tr12)
        thr = float(h["METRIC_THRESHOLD"]) if threshold is None else float(threshold)
        preds = (probs >= thr).astype(int)
        return float(f1_score(y_tr12, preds)), float(thr)


    

    def f1_minus_entropy(CIRS, Labels, model_prob_head, model_logit_head, threshold=None,
                        optimize_threshold=False, entropy_base='e', weights=None):
        f1,thr = f1_on_train12(CIRS, Labels, model_prob_head,
                                threshold=threshold, optimize_threshold=optimize_threshold, weights=weights)
        ent = entropy_on_adaption_logits(model_logit_head, CIRS["ADAPTION"], base=entropy_base)

        return {"f1_train12": f1, "entropy_adaption": ent,"threshold_used":thr, "score_f1_minus_entropy": f1 - ent}

    # --- F1(TRAIN1∪TRAIN2) – Entropy(ADAPTION) ---
    f1ent = f1_minus_entropy(
        CIRS=CIRS,
        Labels=LosLabels,
        model_prob_head=los_model,                # probabilities
        model_logit_head=los_model_entropy,       # logits
        optimize_threshold=True,                  # pick best threshold on Train1+Train2
        entropy_base='e',                         # 'e' for nats; use '2' for bits
        weights=Weights
    )
    if DEBUG:
        print("[F1-Entropy] F1_train12={:.4f}, thr={:.3f}, entropy_adaption={:.4f}, score={:.4f}"
            .format(f1ent["f1_train12"], f1ent["threshold_used"],
                    f1ent["entropy_adaption"], f1ent["score_f1_minus_entropy"]))

    

    alpha_focal = h.get("FOCAL_ALPHA", 0.25)
    gamma_focal = h.get("FOCAL_GAMMA", 1.5)

    # Probabilities (prob-head) per split
    probs_tr1  = predict_probs(los_model, CIRS["TRAIN1"],  Weights["TRAIN1"])
    probs_tr2  = predict_probs(los_model, CIRS["TRAIN2"],  Weights["TRAIN2"])
    probs_test = predict_probs(los_model, CIRS["TEST"],    Weights["TEST"])

    # Logits (logit-head) for ADAPTION (unsupervised regularizers)
    logits_adap = predict_logits(los_model_entropy, CIRS["ADAPTION"])

    mask_tr1 = (np.asarray(Weights["TRAIN1"]) != 0)
    mask_tr2 = (np.asarray(Weights["TRAIN2"]) != 0)

    y_tr1  = np.asarray(LosLabels["TRAIN1"])[mask_tr1].astype(int)
    y_tr2  = np.asarray(LosLabels["TRAIN2"])[mask_tr2].astype(int)
    y_test = np.asarray(LosLabels["TEST"]).astype(int)

    # Concatenate Train1+Train2 (source supervised)
    probs_tr12 = np.concatenate([probs_tr1, probs_tr2])
    y_tr12     = np.concatenate([y_tr1, y_tr2])

    # --- Supervised losses (source & test) on probabilities ---
    bce_train12   = _binary_crossentropy_mean(y_tr12, probs_tr12)
    focal_train12 = _binary_focal_mean      (y_tr12, probs_tr12, gamma=gamma_focal, alpha=alpha_focal)

    bce_test   = _binary_crossentropy_mean(y_test, probs_test)
    focal_test = _binary_focal_mean      (y_test, probs_test, gamma=gamma_focal, alpha=alpha_focal)

    

    # --- Unsupervised regularizers on ADAPTION (use logits for entropy) ---
    entropy_adaption_e    = float(bernoulli_entropy_from_logits(logits_adap, base='e').numpy())
    entropy_adaption_bits = float(bernoulli_entropy_from_logits(logits_adap, base=2).numpy())

    # --- Combined scoring examples ---
    LAMBDA_ENT = float(h.get("LAMBDA_ENTROPY", 1.0))
    combined_ce_ent_train12_adapt   = bce_train12   + LAMBDA_ENT * entropy_adaption_e
    combined_focal_ent_train12_adapt= focal_train12 + LAMBDA_ENT * entropy_adaption_e

    # For reporting both entropy forms on ADAPTION:
    # logits already computed above; reuse them
    adapt_entropy        = float(bernoulli_entropy_from_logits(logits_adap, base='e').numpy())
    adapt_binary_entropy = binary_entropy(tf.math.sigmoid(tf.convert_to_tensor(logits_adap, tf.float32)))



            #@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@  
    #%#%#%#%
    los_model.save(h["save_plots"] / "los_model.keras")
    ae.save(h["save_plots"] / "ae_model.keras")
    adapt_probs = los_model.predict(CIRS["ADAPTION"][..., None], verbose=0).reshape(-1)
    K = max(100, int(0.02 * adapt_probs.size))          # 2% or at least 100 samples
    K = min(K, max(1, adapt_probs.size // 4))           # keep sane upper bound
    topK = np.partition(adapt_probs, -K)[-K:]
    botK = np.partition(adapt_probs,  K)[:K]
    pl_margin_adapt = float(topK.mean() - botK.mean())   # higher is better


   
    H_bits = float(entropy_adaption_bits)


    # prevalence mismatch (use supervised source only: TRAIN1+TRAIN2 with w!=0)
    src_mask_tr1 = (Weights["TRAIN1"] != 0)
    src_mask_tr2 = (Weights["TRAIN2"] != 0)
    y_src = np.concatenate([LosLabels["TRAIN1"][src_mask_tr1], LosLabels["TRAIN2"][src_mask_tr2]]).astype(float)
    y_src_mean = float(np.mean(y_src))
    p_tgt_mean = float(np.mean(adapt_probs))
    prev_gap = abs(p_tgt_mean - y_src_mean)

    # domain confusion (optional)
    chance = 1.0 / float(num_dom)
    val_dom_acc_now = float(history["val_dom_acc"][-1]) if history["val_dom_acc"] else chance
    dom_confusion = 1.0 - abs(val_dom_acc_now - chance) / (1.0 - chance)

    # consistency goodness (optional)
    val_cons = float(val_cons_loss.result()) if h.get("CONS_ENABLE", False) else 0.0
    consistency_good = 1.0 / (1.0 + max(val_cons, 0.0))

    # FINAL score (fix weights; do NOT tune them as hyperparams)
    optuna_score = (
        0.55 * float(val_auroc)
    + 0.25 * float(pl_margin_adapt)
    - 0.10 * H_bits
    - 0.05 * prev_gap
    + 0.05 * dom_confusion
    + 0.05 * consistency_good
    )  


    config = {
                    "SEED": h["SEED"],
                    "LATENT_DIM": h["LATENT_DIM"],
                    "AE_EPOCHS": h["AE_EPOCHS"],
                    "AE_BATCH": h["AE_BATCH"],
                    "AE_PATIENCE": h["AE_PATIENCE"],
                    "BASE_LR": h["BASE_LR"],
                    "GRL_LAMBDA_MAX": h["GRL_LAMBDA_MAX"],
                    "ADAPTION_WITH_LABEL": h["ADAPTION_WITH_LABEL"],
                    "FOCAL_GAMMA": h["FOCAL_GAMMA"],
                    "TRAIN_SIZE": h["TRAIN_SIZE"],
                    "TRAIN1_NAME": h["TRAIN1_NAME"],
                    "TRAIN2_NAME": h["TRAIN2_NAME"],
                    "TEST_NAME": h["TEST_NAME"],
                    "training_time": training_time,
                    "save_plots": h["save_plots"],
                    "CIRS_shapes Train1": CIRS["TRAIN1"].shape,
                    "CIRS_shapes Train2": CIRS["TRAIN2"].shape,
                    "CIRS_shapes Adaption": CIRS["ADAPTION"].shape,
                    "CIRS_shapes Test": CIRS["TEST"].shape,
                    "labels_shapes Train1": LosLabels["TRAIN1"].shape,
                    "labels_shapes Train2": LosLabels["TRAIN2"].shape,
                    "labels_shapes Adaption": LosLabels["ADAPTION"].shape,
                    "labels_shapes Test": LosLabels["TEST"].shape,
                    "Train1 weights": pd.Series(Weights["TRAIN1"]).value_counts().to_dict(),
                    "Train2 weights": pd.Series(Weights["TRAIN2"]).value_counts().to_dict(),
                    "Train12 weights": pd.Series(w_tr).value_counts().to_dict(),
                    "validation weights": pd.Series(w_val).value_counts().to_dict(),
             
                    "PL_WEIGHT": h.get("PL_WEIGHT", "unspecified"),
                    "PL_START_EPOCH": h.get("PL_START_EPOCH", "unspecified"),
                    "PL_UPDATE_EVERY": h.get("PL_UPDATE_EVERY", "unspecified"),
                    "K_EACH": h.get("K_EACH", "unspecified"),
                    "Train1 Domains": pd.Series(Domains["TRAIN1"]).value_counts().to_dict(),
                    "Train2 Domains": pd.Series(Domains["TRAIN2"]).value_counts().to_dict(),
                    "Train12 Domains": pd.Series(d_tr).value_counts().to_dict(),
                    "validation Domains": pd.Series(d_val).value_counts().to_dict(),
                    "LW_REC_START": h["LW_REC_START"],
                    "LW_DOM_START": h["LW_DOM_START"],
                    "LW_LOS_START": h["LW_LOS_START"],
                    "LW_REC_END": h["LW_REC_END"],
                    "LW_DOM_END": h["LW_DOM_END"],
                    "LW_LOS_END": h["LW_LOS_END"],
                    "UNFREEZE_STEP": h["UNFREEZE_STEP"],
                    "METRIC_THRESHOLD": h["METRIC_THRESHOLD"],
                    "FOCAL_ALPHA": h["FOCAL_ALPHA"],



                    
                    "Domains_shapes": {key: Domains[key].shape for key in Domains.keys()},
                    "balanced_dsets_shapes": {key: balanced_dsets[key].shape for key in balanced_dsets.keys()},

                    "Train1Accuracy":TRAIN1_res["accuracy"],
                    "Train1Loss": TRAIN1_res["loss"],
                    "Train1F1Score": TRAIN1_res["f1_score"],
                    "Train1_confusion_matrix": TRAIN1_res["confusion_matrix"],

                    "Train2Accuracy": TRAIN2_res["accuracy"],
                    "Train2Loss": TRAIN2_res["loss"],
                    "Train2F1Score": TRAIN2_res["f1_score"],
                    "Train2_confusion_matrix": TRAIN2_res["confusion_matrix"],

                    "AdaptionAccuracy": ADAPTION_res["accuracy"],
                    "AdaptionLoss": ADAPTION_res["loss"],
                    "AdaptionF1Score": ADAPTION_res["f1_score"],
                    "Adaption_confusion_matrix": ADAPTION_res["confusion_matrix"],

                    "Validation Accuracy_filtered": accuracy_valid,
                    "Validation F1 Score_filtered": f1_valid,
                    "Validation Confusion Matrix_filtered": cm_valid,
                    "Validation Entropy_w0": valid_entropy,

                    
                    "TestAccuracyLastEpoch": TEST_res_last_epoch["accuracy"],
                    "TestLossLastEpoch": TEST_res_last_epoch["loss"],
                    "TestF1ScoreLastEpoch": TEST_res_last_epoch["f1_score"],
                    "Test_confusion_matrix": TEST_res_last_epoch["confusion_matrix"],
                    
                    
                    "TestLoss": TEST_res["loss"],
                    "TestAccuracy": TEST_res["accuracy"],
                    "TestF1Score": TEST_res["f1_score"],
                    "Test_confusion_matrix": TEST_res["confusion_matrix"],

                    "AccuracyTrain_filtered": accuracy_train,
                    "F1Train_filtered": f1_train,
                    "ConfusionMatrixTrain_filtered": cm_train,

                    "EntropyTrain_w0": train_entropy,
                    "F1EntropyTrain": f1_entropy_train,
                    
                    
                    
                    "Validation F1 (entropy)": f1_entropy_valid,
                    "TEST_F1_Score":TEST_res["f1_score"],
                    "Validation_F1_Score": f1_valid,

                    
                    "TEST_Confusion_Matrix": TEST_res["confusion_matrix"],
                    "Validation_Confusion_Matrix": cm_valid,
                    
                    "score_f1_minus_entropy": f1ent["score_f1_minus_entropy"],
                    "f1_train12": f1ent["f1_train12"],
                    "threshold_used": f1ent["threshold_used"],
                    "entropy_adaption": f1ent["entropy_adaption"],
                    "bce_train12": bce_train12,
                    "focal_train12": focal_train12,
                    # Test
                    "bce_test": bce_test,
                    "focal_test": focal_test,
                    # Target (Adaption) unsupervised regularizers
                    "entropy_adaption_e": entropy_adaption_e,
                    "entropy_adaption_bits": entropy_adaption_bits,
                   
                    # Combined objective samples
                    "lambda_entropy": LAMBDA_ENT,
                    "combined_ce_plus_lambda_entropy": combined_ce_ent_train12_adapt,
                    "combined_focal_plus_lambda_entropy": combined_focal_ent_train12_adapt,
                    "validation_binary_entropy": validation_binary_entropy,
                    "f1_entropy_valid": f1_entropy_valid,
                    "validation_dsc": validation_dsc.numpy().item(),
                    "validation_dice_loss": validation_dice_loss.numpy().item(),
                    "validation_bce_dice": validation_bce_dice.numpy().item(),
                    "validation_bce": validation_bce.numpy().item(),
                    "validation_tp": validation_tp.numpy().item(),
                    "validation_tn": validation_tn.numpy().item(),
                    "validation_focal_tversky": validation_focal_tversky.numpy().item(),
                    "validation_binary_focal_crossentropy": validation_binary_focal_crossentropy.numpy().item(),
                    "train_binary_entropy": train_binary_entropy,
                    "train_dsc": train_dsc.numpy().item(),
                    "train_dice_loss": train_dice_loss.numpy().item(),
                    "train_bce_dice": train_bce_dice.numpy().item(),
                    "train_bce": train_bce.numpy().item(),
                    "train_tp": train_tp.numpy().item(),
                    "train_tn": train_tn.numpy().item(),
                    "train_focal_tversky": train_focal_tversky.numpy().item(),
                    "train_bce_focal_crossentropy": train_binary_focal_crossentropy.numpy().item(),
                    "adapt_entropy": adapt_entropy,
                    "adapt_binary_entropy": adapt_binary_entropy,
                    "optuna_score": optuna_score,
                    "train_balanced_accuracy_score":train_balanced_accuracy_score,
                    "validation_balanced_accuracy_score":validation_balanced_accuracy_score,
                    "train_mean_entropy":train_mean_entropy,
                    "train_weighted_bce_logits": train_weighted_bce_logits,
                    "validation_weighted_bce_logits": validation_weighted_bce_logits



                }
    config = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in config.items()}
    for k, v in config.items():
        if isinstance(v, str) and v.replace('.', '', 1).isdigit():
            config[k] = float(v) if '.' in v else int(v)

   
    
    save_to_excel(config, h["save_plots"])
    #$#$#$#
    encoder = tf.keras.Model(
        inputs  = ae.input,
        outputs = ae.get_layer("latent_vector").output,
        name    = "encoder"
    )

    # ---- decoder ----------------------------------------------------------------
    # 1) grab the nested decoder Model by name
    decoder_layer = ae.get_layer("reconstruction")   # THIS is a tf.keras.Model
    # 2) re-wrap on fresh Input
    latent_in  = tf.keras.Input(shape=(h["LATENT_DIM"],), name="latent_in")
    recon_out  = decoder_layer(latent_in)
    decoder    = tf.keras.Model(latent_in, recon_out, name="decoder")

    
    # %%
    encoder.predict(CIRS["TEST"][...,None])
    ind = np.where(LosLabels["TEST"] == 0.0)
    if DEBUG:
        print("Number of LOS samples in TEST set:", len(ind[0]))
    plt.plot(encoder.predict(CIRS["TEST"][...,None])[ind[0][0:50]],color='blue', label='NLOS samples')

    encoder.predict(CIRS["TEST"][...,None])
    ind = np.where(LosLabels["TEST"] == 1.0)
    if DEBUG:
        print("Number of LOS samples in TEST set:", len(ind[0]))
    plt.plot(encoder.predict(CIRS["TEST"][...,None])[ind[0][0:50]],color='red', label='LOS samples')
    if DEBUG:
        print("Plotting umap 1")

    plot_latent_umap(encoder, CIRS, Domains,h["save_plots"])

    if DEBUG:
        print("Plotting umap 2")
    # Call this:
    plot_latent_umap2(encoder, CIRS, Domains,h["save_plots"])

    # %%
   

    # How many adaptation samples have non-zero weight?
    w_adapt = Weights["ADAPTION"]
    if DEBUG:
        print("ADAPTION_WITH_LABEL =", h["ADAPTION_WITH_LABEL"])
        print("Unique weights on ADAPTION labels:", np.unique(w_adapt))
        print("Count of non-zero weights:", np.count_nonzero(w_adapt))




    # How many adaptation samples have non-zero weight?
    w_adapt = Weights["ADAPTION"]
    if DEBUG:
        print("ADAPTION_WITH_LABEL =", h["ADAPTION_WITH_LABEL"])
        print("Unique weights on ADAPTION labels:", np.unique(w_adapt))
        print("Count of non-zero weights:", np.count_nonzero(w_adapt))
    ##############################
    ############################


    # 3) Prepare test data
    X_test      = CIRS["TEST"][..., None]    # shape: (N, seq_len, 1)
    y_test_los  = LosLabels["TEST"].astype(int) # shape: (N,)

    # 4) Evaluate
    res = los_model.evaluate(X_test, y_test_los, return_dict=True)
    if DEBUG:
        print(f"LOS‑only test loss:     {res['loss']:.4f}")
        print(f"LOS‑only test accuracy: {res['accuracy']:.4f}")

    # 5) Predict and then compute confusion/F1
    y_pred_los     = los_model.predict(X_test).flatten()        # P(NLOS)

    plot_encoded_signals(ae,decoder,encoder, CIRS, LosLabels, h["save_plots"])


    ######################################
    ##################################
    plot_predicted_vs_real(y_pred_los, y_test_los, h["save_plots"])
   








    probe_layer_names = ["latent_vector", "latent_cls", "attention_gated","los_prob"]
    probe_outputs = [ae.get_layer(name).output
                        for name in probe_layer_names]
    probe_model   = tf.keras.Model(inputs=ae.input,
                            outputs=probe_outputs)

    probe_ae(ae, CIRS,LosLabels, h["save_plots"])


    evaluate_model_performance(probe_model, CIRS, LosLabels, h["save_plots"],"TRAIN1")
    evaluate_model_performance(probe_model, CIRS, LosLabels, h["save_plots"],"TRAIN2")
    evaluate_model_performance(probe_model, CIRS, LosLabels, h["save_plots"],"ADAPTION")
    evaluate_model_performance(probe_model, CIRS, LosLabels, h["save_plots"],"TEST")

    plot_confusion_matrix(los_model,h["save_plots"], CIRS, LosLabels,"TRAIN1",h["METRIC_THRESHOLD"])
    plot_confusion_matrix(los_model,h["save_plots"], CIRS, LosLabels,"TRAIN2",h["METRIC_THRESHOLD"])
    plot_confusion_matrix(los_model,h["save_plots"], CIRS, LosLabels,"ADAPTION",h["METRIC_THRESHOLD"])
    plot_confusion_matrix(los_model,h["save_plots"], CIRS, LosLabels,"TEST",h["METRIC_THRESHOLD"])
   

   
    TEXT_label="ADAPTION"  # or "TRAIN2", "ADAPTION", "TEST"
    CIRS_test_LOS= CIRS[TEXT_label][LosLabels[TEXT_label]==0][..., None]
    CIRS_test_NLOS= CIRS[TEXT_label][LosLabels[TEXT_label]==1][..., None]
    # get every test example’s predicted LOS-probability
    all_X = np.concatenate([CIRS_test_LOS, CIRS_test_NLOS], axis=0)
    all_y = np.concatenate([np.zeros(len(CIRS_test_LOS)),
                            np.ones(len(CIRS_test_NLOS))])
    all_probs = probe_model.predict(all_X)[-1].squeeze()
    # [-1] grabs the los_logits output; shape (N,1) → .squeeze() → (N,)

    # threshold at 0.5
    all_pred = (all_probs >= 0.5).astype(int)
    if DEBUG:

        print(classification_report(all_y, all_pred, target_names=["LOS","NLOS"]))
        print(f"Confusion matrix:{TEXT_label}\n", confusion_matrix(all_y, all_pred))
        print("ROC AUC:    ", roc_auc_score(all_y, all_probs))



    

    return ae, ae_history, accuracy_valid, f1_valid, cm_valid, training_time,f1ent,val_auroc, optuna_score





