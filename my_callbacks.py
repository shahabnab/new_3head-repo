from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

class ProgressiveUnfreeze(Callback):
    """Unfreezes one Conv1D block every *step_epochs* epochs."""
    def __init__(self, encoder: models.Model, step_epochs: int = 5):
        super().__init__()
        # grab all your Conv1D layers (in order)
        self.blocks = [l for l in encoder.layers if isinstance(l, tf.keras.layers.Conv1D)]
        self.step   = step_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # how many blocks to unfreeze so far
        idx = epoch // self.step
        # unfreeze the last `idx` blocks
        for i, blk in enumerate(self.blocks):
            blk.trainable = (i >= len(self.blocks) - idx)

class TestMetrics(Callback):
            def __init__(self, x, y_list, w_list=None):
                super().__init__()
                self.x = x
                self.y = y_list
                self.w = w_list

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                results = self.model.evaluate(
                    x=self.x,
                    y=self.y,
                    sample_weight=self.w,
                    verbose=0,
                    return_dict=True
                )
                for name, value in results.items():
                    logs[f"test_{name}"] = value

 # ---- GRL schedule (same shape as before) ----


class GRLSchedule(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, max_epochs, lambda_min=0.0, lambda_max=1.0, peak_frac=1.0):
        super().__init__()
        self.grl_layer  = grl_layer
        self.max_epochs = int(max(1, max_epochs))
        self.lmin       = tf.convert_to_tensor(lambda_min, tf.float32)
        self.lmax       = tf.convert_to_tensor(lambda_max, tf.float32)
        # reach λ_max at fraction of training (clamped to (0,1])
        self.peak_frac  = float(min(1.0, max(1e-6, peak_frac)))

    def on_epoch_begin(self, epoch, logs=None):
        p = tf.cast(epoch, tf.float32) / tf.cast(self.max_epochs - 1 + 1e-7, tf.float32)
        q = tf.minimum(1.0, p / self.peak_frac)  # ramp until peak_frac, then hold
        lam = 0.5 * (1.0 - tf.cos(tf.constant(np.pi, tf.float32) * q))
        lam = self.lmin + (self.lmax - self.lmin) * lam
        self.grl_layer.hp_lambda.assign(lam)



class EarlyStoppingOnGap(Callback):
        """Stop when train/val accuracy gap is minimal after warm-up."""
        def __init__(
            self,
            monitor_train='accuracy',
            monitor_val='val_accuracy',
            min_epochs=5,
            patience=2,
            restore_best_weights=True
        ):
            super().__init__()
            self.mon_tr = monitor_train
            self.mon_va = monitor_val
            self.min_epochs = min_epochs
            self.patience = patience
            self.restore_best_weights = restore_best_weights
            self.best_gap = np.inf
            self.best_weights = None
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            tr = logs.get(self.mon_tr)
            va = logs.get(self.mon_va)
            if tr is None or va is None or (epoch + 1) < self.min_epochs:
                return
            gap = abs(tr - va)
            if gap < self.best_gap:
                self.best_gap = gap
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    print(
                        f"\nStopping at epoch {epoch + 1}. "
                        f"Best gap = {self.best_gap:.4f}."
                    )
class SimpleEarlyStopping(Callback):
    """
    Stop training when `val_los_acc` has not improved for `patience` epochs.
    Does NOT restore any weights—just halts.
    """
    def __init__(self, monitor="val_los_acc", patience=10):
        super().__init__()
        self.monitor  = monitor
        self.patience = patience
        self.best     = -float("inf")
        self.wait     = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            # nothing to do if metric isn't in logs
            return

        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            print(
                f"\nEarly stopping at epoch {epoch+1}: "
                f"no improvement in {self.monitor} for {self.patience} epochs "
                f"(best was {self.best:.4f})."
            )
            # flag to break out of your manual loop
            self.model.stop_training = True    
            


class BestAccSaver:
            """
            Saves best weights by validation accuracy and always saves last-epoch weights.
            """
            def __init__(self, model, save_dir,
                        best_name="best_val_weights.weights.h5",
                        last_name="last_epoch_weight.weights.h5"):
                self.model = model
                self.save_dir = save_dir
                os.makedirs(save_dir, exist_ok=True)
                self.fp_best = os.path.join(save_dir, best_name)
                self.fp_last = os.path.join(save_dir, last_name)
                self.best = -np.inf
                self.best_epoch = -1

            def on_epoch_end(self, epoch_idx: int, val_acc: float):
                # Save last-epoch weights (overwrite each epoch)
                self.model.save_weights(self.fp_last)
                # Save best-by-accuracy
                if val_acc > self.best:
                    self.best = val_acc
                    self.best_epoch = epoch_idx
                    self.model.save_weights(self.fp_best)
                    print(f"[BestAccSaver] New BEST val_acc={val_acc:.4f} at epoch {epoch_idx+1}; saved -> {self.fp_best}")

            def on_train_end(self):
                print(
                    f"[BestAccSaver] Training finished. "
                    f"Best val_acc={self.best:.4f} (epoch {self.best_epoch+1 if self.best_epoch>=0 else 'N/A'}). "
                    f"Last-epoch weights -> {self.fp_last}; best -> {self.fp_best}"
                )            




class WarmupThenCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, alpha=0.1):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        # cosine after warmup
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=max(1, int(total_steps - int(warmup_steps))),
            alpha=alpha
        )

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        # linear warmup
        warm_lr = self.base_lr * (step_f + 1.0) / tf.maximum(1.0, self.warmup_steps)
        # cosine after warmup
        cos_lr  = self.cosine(tf.maximum(0.0, step_f - self.warmup_steps))
        return tf.where(step_f < self.warmup_steps, warm_lr, cos_lr)
