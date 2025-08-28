import pandas as pd
import gdown    
import os
import pickle
from my_cir_processing import cutting_cir
import tensorflow as tf
import numpy as np
def shuffle_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
   
    return (
        df.sample(frac=1.0, random_state=seed)  # permute rows

    )


def download_dataset(datasets_names,SEED):

    ############################## Downloading the datasets ##########################################
    if not os.path.exists("data"):
        os.makedirs("data")
    # Download TUall
    if not os.path.exists("data/TUall.pickle"):
        print("Downloading TUall dataset...")
        gdown.download(id="1R2GOGED6jzLU8I35lu5SHT1jlpCmqk7R", output="data/TUall.pickle", quiet=False)
    # Download IOT_PT1
    if not os.path.exists("data/IOT_PT1.pickle"):
        print("Downloading IOT_PT1 dataset...")
        gdown.download(id="1Xil7h9fHvaFGIE2nWpWUXOIAg1us9Wlo", output="data/IOT_PT1.pickle", quiet=False)
    if not os.path.exists("data/IOT_PT2.pickle"):
    # Download IOT_PT2
        print("Downloading IOT_PT2 dataset...")
        gdown.download(id="1mgOcYSp9BjOqDgp4GYiaQy7Au0pEEsxO", output="data/IOT_PT2.pickle", quiet=False)

    # Download Office
    if not os.path.exists("data/Office.pickle"):
        print("Downloading Office dataset...")
        gdown.download(id="1QhLyo9_4pSfvkXhyJ5DrC4P2qZcf9OdV", output="data/Office.pickle", quiet=False)
############################## using them in dataframes ##########################################


    with open("data/TUall.pickle", "rb") as f:
        TUall = pickle.load(f)


    with open("data/IOT_PT1.pickle", "rb") as f:
        IOT_PT1 = pickle.load(f)


    with open("data/IOT_PT2.pickle", "rb") as f:
        IOT_PT2 = pickle.load(f)

    with open("data/Office.pickle", "rb") as f:
        Office = pickle.load(f)
    #concatenating the IOT_PT1 and IOT_PT2 datasets

    IOT = pd.concat([IOT_PT1[["label", "CIR_amp","Sensor rng","sensor rssi","sensor fp_power","camera rng"]], IOT_PT2[["label", "CIR_amp","Sensor rng","sensor rssi","sensor fp_power","camera rng"]]], axis=0)

    IOT["label"] = IOT["label"].astype(int)
    IOT.rename(columns={"label": "Label"}, inplace=True)
    #chanigng the LOS and NLOS labels to be 0 and 1 in TU dataset
    TUall["CIR_amp"] = TUall["CIR_amp"].apply(lambda x: np.sqrt(x) / 101)
    TU=TUall[TUall["sensor los"].isin({"los", "nlos"})]
    TU["Label"]=TUall["sensor los"]
    TU["Label"]=TU["Label"].map({"los": 0, "nlos": 1})


    TU_cuts=cutting_cir(TU["CIR_amp"].to_numpy(), windows_size=500)
    cuts_list_TU = [row for row in TU_cuts]
    TU["CIR_amp"] = cuts_list_TU

    Office.rename(columns={"label": "Label"}, inplace=True)
    Office["Label"]=Office["Label"].astype(int)
    print("#################################################")
    print("#################################################")
    print("LOS and NLOS distribution in IOT dataset:",IOT["Label"].value_counts())
    print("#################################################")
    print("LOS and NLOS distribution in TU dataset:",TU["Label"].value_counts())
    print("#################################################")
    print("LOS and NLOS distribution in Office dataset:",Office["Label"].value_counts())
    print("#################################################")
    print("#################################################")

    IOT   = shuffle_df(IOT,   seed=SEED)
    TU   = shuffle_df(TU,   seed=SEED)
    Office = shuffle_df(Office, seed=SEED)


    dfs = {
        "IOT": IOT,
        "TU": TU,
        "Office": Office
    }
    train1=dfs[datasets_names[0]]
    train2=dfs[datasets_names[1]]
    test=dfs[datasets_names[2]]
    print("#################################################")
    print("datasets are as follows:")
    print("Train1:", datasets_names[0])
    print("Train2:", datasets_names[1])
    print("Test & Adaption:", datasets_names[2])
    print("#################################################")
    #creating the datasets dictionary
    #creating train1, train2 and test datasets
    datasets={
        "train1": train1,
        "train2": train2,
        "test_adaption": test
    }

    return datasets

def slicing_dts(datasets, save_path, datasets_names, dt_rules, tr_size, SEED):
    Train1, Train2, test_adaption = (
        datasets["train1"], datasets["train2"], datasets["test_adaption"]
    )

    # ── 1) convenience masks ───────────────────────────────────────────────
    def los(df):  return df[df["Label"] == 0]
    def nlos(df): return df[df["Label"] == 1]

    # ── 2) report sizes -----------------------------------------------------
    for name, df in zip(datasets_names, [Train1, Train2, test_adaption]):
        print(f"{name}: NLOS={len(nlos(df))}  LOS={len(los(df))}")

    # ── 3) TRAIN1 / TRAIN2  -------------------------------------------------
    TRAIN1 = pd.concat([
        nlos(Train1).sample(min(tr_size, len(nlos(Train1))), random_state=SEED),
        los(Train1).sample(min(tr_size, len(los(Train1))),  random_state=SEED)
    ], ignore_index=True   # drop old row labels
    )

    TRAIN2 = pd.concat([
        nlos(Train2).sample(min(tr_size, len(nlos(Train2))), random_state=SEED),
        los(Train2).sample(min(tr_size, len(los(Train2))),  random_state=SEED)
    ], ignore_index=True
    )

    # ── 4) ADAPTION  --------------------------------------------------------
    ad_nlos = nlos(test_adaption).sample(min(tr_size, len(nlos(test_adaption))),
                                         random_state=SEED)
    ad_los  = los(test_adaption).sample(min(tr_size, len(los(test_adaption))),
                                        random_state=SEED)
    ADAPTION = pd.concat([ad_nlos, ad_los])              # keep original indices!
    #ADAPTION["Label"]=-1
    # drop **be
    # fore** we reset_index, so labels match
    TEST_REMAINED = test_adaption.drop(ADAPTION.index)
    



    # ── 5) TEST  -----------------------------------------------------------
    test_nlos = nlos(TEST_REMAINED).sample(
        min(1000, len(nlos(TEST_REMAINED))), random_state=SEED)
    test_los  = los(TEST_REMAINED).sample(
        min(1000, len(los(TEST_REMAINED))), random_state=SEED)

    TEST = pd.concat([test_nlos, test_los], ignore_index=True
           )
    TRAIN1   = shuffle_df(TRAIN1,   seed=SEED)
    TRAIN2   = shuffle_df(TRAIN2,   seed=SEED)
    ADAPTION = shuffle_df(ADAPTION, seed=SEED)
    TEST = shuffle_df(TEST, seed=SEED)


    #sampling 1000 samples from the remaining NLOS and LOS samples in Office dataset for testing
    print("Number of NLOS and LOS samples in TU dataset after slicing:")
    print("Datasets for training:\n")
    print(f"{datasets_names[0]} training 1 shape: ",TRAIN1.shape)
    print(f"{datasets_names[1]} training 2 shape: ",TRAIN2.shape)
    print(f"{datasets_names[2]} adaption shape: ",ADAPTION.shape)
    print(f"{datasets_names[2]} dataset test",TEST.shape)

    #Normalizing the sensor range

    #checked
    balanced_dts={
        dt_rules[0]: TRAIN1,
        dt_rules[1]: TRAIN2,
        dt_rules[2]: ADAPTION,
        dt_rules[3]: TEST
    }

    return balanced_dts


""" def make_dataset(
    X, d, l, w,
    batch, num_dom,
    *,
    seed=42,
    shuffle_buf=4096,
    split="train",                 # "train" | "val" | "test"
    pl_mode="hide",                # "hide" | "use"
    pl_domain_id=2                 # domain id of the target/adaption split
):
   
    X = X.astype("float32")
    d = d.astype("int32")
    l = l.astype("float32")
    w = w.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X, d, l, w))

    def _prep(sig, dom, lbl, w_los):
        # (T,) -> (T,1)
        if tf.rank(sig) == 1:
            sig = tf.expand_dims(sig, -1)
        sig = tf.cast(sig, tf.float32)

        y_rec = sig
        y_dom = tf.one_hot(dom, depth=num_dom)

        y_los = tf.reshape(tf.cast(lbl, tf.float32), (1,))
        sw_los = tf.cast(w_los, tf.float32)

        sw_rec = tf.constant(1.0, tf.float32)
        sw_dom = tf.constant(1.0, tf.float32)

        is_target = tf.equal(dom, tf.cast(pl_domain_id, dom.dtype))

        # train/val masking policy
        if split != "test":
            if pl_mode == "hide":
                # Hide target labels, zero their classifier weight
                y_los  = tf.where(is_target, -1.0, y_los)
                sw_los = tf.where(is_target,  0.0, sw_los)
            # pl_mode == "use": keep labels/weights as given (e.g., pseudo labels)

        return sig, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los)

    ds = (
        ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
          .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
    )
    return ds """

def make_dataset(
    X, d, l, w,
    batch, num_dom,
    *,
    seed=42,
    shuffle_buf=4096,
    split="train",                 # "train" | "val" | "test"
    pl_mode="hide",                # "hide" | "use"
    pl_domain_id=2                 # target/adaptation domain id
):
    """
    Build a tf.data pipeline with batched map, static shapes, and version-safe optimizations.
    - Domain IDs are int64 to avoid XLA S32/S64 mismatches under jit_compile=True.
    - Batch BEFORE map to enable vectorized ops and kernel fusion.
    - Prefetch to GPU when available to overlap H2D with compute.
    """
    import tensorflow as tf

    # ---- host-side normalization (do once) ----
    X = X.astype("float32")
    if X.ndim == 2:          # (N,T) -> (N,T,1) once on host (cheaper than per-element in map)
        X = X[..., None]
    d = d.astype("int64")    # int64 indices play nicer with XLA
    l = l.astype("float32")
    w = w.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X, d, l, w))

    # Optional: cache in RAM if it fits (uncomment if memory allows)
    # ds = ds.cache()

    ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=True)  # static shapes -> better fusion/JIT

    # ---- batched map (vectorizable) ----
    def _prep_batched(sig, dom, lbl, w_los):
        # sig: (B,T,1) float32
        y_rec = sig
        y_dom = tf.one_hot(dom, depth=num_dom, dtype=tf.float32)  # (B,num_dom)

        y_los  = tf.expand_dims(tf.cast(lbl,  tf.float32), -1)    # (B,1)
        sw_los = tf.cast(w_los, tf.float32)                       # (B,)

        sw_rec = tf.ones_like(sw_los, dtype=tf.float32)           # (B,)
        sw_dom = tf.ones_like(sw_los, dtype=tf.float32)           # (B,)

        if split != "test" and pl_mode == "hide":
            is_target = tf.equal(dom, tf.cast(pl_domain_id, dom.dtype))  # (B,)
            y_los  = tf.where(is_target[:, None], -1.0, y_los)           # mask labels
            sw_los = tf.where(is_target,          0.0,  sw_los)          # zero weights

        return sig, (y_rec, y_dom, y_los), (sw_rec, sw_dom, sw_los)

    ds = ds.map(_prep_batched, num_parallel_calls=tf.data.AUTOTUNE)

    # ---- prefetch to device when available (overlap input with compute) ----
    try:
        from tensorflow.data.experimental import copy_to_device, prefetch_to_device
        ds = ds.apply(copy_to_device('/GPU:0')).apply(prefetch_to_device(1))
    except Exception:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    # ---- version-safe tf.data optimizations ----
    opts = tf.data.Options()

    # Determinism toggle (API name differs across TF versions)
    if hasattr(opts, "deterministic"):
        opts.deterministic = False
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = False

    eo = opts.experimental_optimization

    # Enable commonly-available flags if present in this TF build
    for flag in [
        "autotune", "autotune_buffers",
        "map_fusion", "map_parallelization",
        "map_and_batch_fusion",
        "filter_fusion", "noop_elimination",
        "parallel_batch", "slack",
    ]:
        if hasattr(eo, flag):
            setattr(eo, flag, True)

    # Handle map_vectorization API differences across versions
    mv = getattr(eo, "map_vectorization", None)
    if isinstance(mv, bool):
        eo.map_vectorization = True
    elif mv is not None and hasattr(mv, "enabled"):
        mv.enabled = True
    # else: not available → skip silently

    ds = ds.with_options(opts)
    return ds
