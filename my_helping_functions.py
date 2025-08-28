import os
import random
import numpy as np
import tensorflow as tf
import gdown
import pickle
import pandas as pd

""" def set_seed(seed: int, deterministic_threads: bool = True, cleanup=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic_threads:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.backend.clear_session() """


def set_seed(seed: int, *, enable_tf_op_determinism: bool = True, verbose: bool = True) -> None:
    """
    Reproducible runs on GPU without disabling parallelism.

    What it does:
      • Seeds Python, NumPy, and TensorFlow RNGs.
      • (Optionally) enables deterministic TF ops (cuDNN) when available.
      • DOES NOT force thread counts to 1 (keeps performance).

    Tips (do these in your entry script BEFORE importing TensorFlow):
      os.environ["TF_DETERMINISTIC_OPS"] = "1"
      os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
      # optional:
      # os.environ["PYTHONHASHSEED"] = str(seed)  # only effective at process start

    For tf.data pipelines, keep order deterministic:
      ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=False)
      ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    """
    import random
    import numpy as np
    import tensorflow as tf

    # RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prefer TF's built-in deterministic switch (TF ≥ 2.12)
    if enable_tf_op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # older TF versions: rely on the env vars set before import
            pass

    # Do NOT force single-thread execution — keep resources available.
    # If you ever want capped-but-parallel threads, set fixed numbers, e.g.:
    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(2)

    if verbose:
        print(f"[seed] {seed}  | TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")


   





def take_labels(datasets):
    datasets_roles = ["TRAIN1","TRAIN2","ADAPTION","TEST"]


    df_tr1  = datasets[datasets_roles[0]]
    df_tr2  = datasets[datasets_roles[1]]
    df_dt   = datasets[datasets_roles[2]]
    df_test = datasets[datasets_roles[3]]

    tr1_lbl = df_tr1["Label"].astype(np.int32)
    tr2_lbl = df_tr2["Label"].astype(np.int32)
    ad_lbl = df_dt["Label"].astype(np.int32)
    test_lb = df_test["Label"].astype(np.int32)
    Labels = {
        "role": [],
        "data": []
    }
    for name, data in zip(datasets_roles, [tr1_lbl, tr2_lbl, ad_lbl, test_lb]):
        Labels["role"].append(name)
        Labels["data"].append(data)

    return  dict(zip(Labels["role"], Labels["data"]))





def take_domains(datasets):
    Domains={
        "role": [],
        "data": []
    }
    datasets_roles = ["TRAIN1","TRAIN2","ADAPTION","TEST"]
    dom_tr1  = np.zeros(len(datasets[datasets_roles[0]]))
    dom_tr2  = np.ones(len(datasets[datasets_roles[1]]))
    dom_adap   = np.ones(len(datasets[datasets_roles[2]]))*2
    dom_test = np.ones(len(datasets[datasets_roles[3]]))*2


    for name, data in zip(datasets_roles, [dom_tr1, dom_tr2, dom_adap, dom_test]):
        Domains["role"].append(name)
        Domains["data"].append(data)
    return dict(zip(Domains["role"], Domains["data"]))



def take_weights(datasets,adapt_size=0):

    datasets_roles = ["TRAIN1","TRAIN2","ADAPTION","TEST"]
    w_tr1  = np.ones(len(datasets[datasets_roles[0]]))
    w_tr2  = np.ones(len(datasets[datasets_roles[1]]))
    w_adap   = np.zeros(len(datasets[datasets_roles[2]]))
    w_test = np.ones(len(datasets[datasets_roles[3]]))
    weights={
        "role": [],
        "data": []
    }
    if adapt_size > 0:
        idx0=np.where(datasets["ADAPTION"]["Label"]==0)[0]
        idx1=np.where(datasets["ADAPTION"]["Label"]==1)[0]
        chosen0 = np.random.choice(idx0, size=adapt_size, replace=False)
        chosen1 = np.random.choice(idx1, size=adapt_size, replace=False)
        w_adap[chosen0] = 1
        w_adap[chosen1] = 1


    for name, data in zip(datasets_roles, [w_tr1, w_tr2, w_adap, w_test]):
        weights["role"].append(name)
        weights["data"].append(data)
    return dict(zip(weights["role"], weights["data"]))

def take_rng(datasets):
    datasets_roles = ["TRAIN1","TRAIN2","ADAPTION","TEST"]
    df_tr1  = datasets[datasets_roles[0]]
    df_tr2  = datasets[datasets_roles[1]]
    df_dt   = datasets[datasets_roles[2]]
    df_test = datasets[datasets_roles[3]]

    tr1_lbl = df_tr1["camera rng"].astype(np.float32)
    tr2_lbl = df_tr2["camera rng"].astype(np.float32)
    ad_lbl = df_dt["camera rng"].astype(np.float32)
    test_lb = df_test["camera rng"].astype(np.float32)
    Labels = {
        "role": [],
        "data": []
    }
    for name, data in zip(datasets_roles, [tr1_lbl, tr2_lbl, ad_lbl, test_lb]):
        Labels["role"].append(name)
        Labels["data"].append(data)

    return  dict(zip(Labels["role"], Labels["data"]))
def train_valid_split(N,train_ratio,random_seed):
    np.random.seed(random_seed)
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size = int(train_ratio * N)  # 80% for training
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    return train_indices, valid_indices


def print_distribution(data_dis):
    unique, counts = np.unique(data_dis, return_counts=True)
    for val, count in zip(unique, counts):
            print(f"{val}: {count}")


def save_to_excel(config, excel_path):
    



    df_cfg = pd.DataFrame([config])
    save_path=excel_path/"report.xlsx"
    df_cfg.to_excel(save_path, index=False)
    print(f"Configuration written to {save_path}")

 


from sklearn.metrics import confusion_matrix, f1_score
def predict_los_only(CIRS, labels,lb_rule,Los_model,WEIGHTS=None):
   
    # 3) Prepare test data
    X  = CIRS[lb_rule][..., None]    # shape: (N, seq_len, 1)
    y  = labels[lb_rule].astype(int) # shape: (N,)
    if WEIGHTS is not None and lb_rule != "ADAPTION":
        mask=WEIGHTS[lb_rule] != 0.0
        X = X[mask]
        y = y[mask]
    



    # 4) Evaluate
    res = Los_model.evaluate(X, y, return_dict=True)
    print(f"LOS‑only test loss:     {res['loss']:.4f}")
    print(f"LOS‑only test accuracy: {res['accuracy']:.4f}")

    # 5) Predict and then compute confusion/F1
    y_prob     = Los_model.predict(X).flatten()        # P(NLOS)
    y_pred_bin = (y_prob >= 0.5).astype(int)

    print(f"Confusion matrix {lb_rule}:\n", confusion_matrix(y, y_pred_bin))

    print(f"accuracy: {res['accuracy']:.4f}")
    print(f"LOS F1 Score: {f1_score(y, y_pred_bin):.4f}")
    res= {
        "loss": res["loss"],
        "accuracy": res["accuracy"],
        "f1_score": f1_score(y, y_pred_bin),
        "confusion_matrix": confusion_matrix(y, y_pred_bin)
    }
    return res
