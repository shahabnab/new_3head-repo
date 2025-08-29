from pathlib import Path

import optuna
from optuna.trial import TrialState
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # quieter logs


import tensorflow as tf
from tensorflow.keras import mixed_precision


for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)









import matplotlib
import sys, argparse
matplotlib.use('Agg', force=True)

import gc;

# If you want mixed precision, control it here once:
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Import only what you use
from defined_functions import (
    download_dataset, slicing_dts, CIR_pipeline, take_labels,
    take_domains, take_weights, take_rng, set_seed, train_ae
)

# ---------------- utilities ----------------
def combine_trial_reports(scenario_dir: Path):
    paths = list(scenario_dir.glob("trial_*/report.xlsx"))
    if not paths:
        print(f"ℹ️  No report.xlsx under {scenario_dir}. Skipping.")
        return
    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p)
            df["trial_id"] = p.parent.name
            frames.append(df)
        except Exception as e:
            print(f"⚠️  Could not read {p}: {e}")
    if not frames:
        print(f"ℹ️  No readable report files under {scenario_dir}.")
        return
    combined = pd.concat(frames, ignore_index=True)
    out_xlsx = scenario_dir / "all_reports.xlsx"
    combined.to_excel(out_xlsx, index=False)
    try:
        combined.to_parquet(scenario_dir / "all_reports.parquet", index=False)
    except Exception as e:
        print(f"⚠️  Parquet write failed: {e}")
    print(f"✅  Combined {len(paths)} files → {out_xlsx}")

# scenarios
dt_names = [
    ["IOT", "TU", "Office"],
    ["TU", "Office", "IOT"],
    ["IOT", "Office", "TU"],
]
train_sizes = [8000, 7000, 6500]
enable_pseudo_labeling = False

for DATASET_NAMES, TRAIN_SIZE in zip(dt_names, train_sizes):
    SEED = 42
    DATASET_ROLES = ["TRAIN1", "TRAIN2", "ADAPTION", "TEST"]
    ADAPTION_WITH_LABEL = 0
    # Prefer a descriptive scenario folder:
    scenario_tag = f"{DATASET_NAMES[0]}_{DATASET_NAMES[1]}_to_{DATASET_NAMES[2]}"
    SAVE_PLOTS_ROOT = Path(scenario_tag)
    SAVE_PLOTS_ROOT.mkdir(exist_ok=True)
    WHOLE_RES_XLSX = "whole_res.xlsx"
    CONFIG_ATTR = "config"

    tf.random.set_seed(SEED)
    set_seed(SEED)

    print("📥  Downloading + slicing datasets …")
    raw_datasets   = download_dataset(DATASET_NAMES, SEED)
    balanced_dsets = slicing_dts(
        raw_datasets,
        save_path=f"res_SEED/{SEED}/sample_{ADAPTION_WITH_LABEL}",
        datasets_names=DATASET_NAMES,
        dt_rules=DATASET_ROLES,
        tr_size=TRAIN_SIZE,
        SEED=SEED,
    )

    CIRS    = CIR_pipeline(balanced_dsets)
    Labels  = take_labels(balanced_dsets)
    Domains = take_domains(balanced_dsets)
    Weights = take_weights(balanced_dsets, ADAPTION_WITH_LABEL)
    RNG     = take_rng(balanced_dsets)

    print("✅  Dataset pipeline ready.\n")

    def objective(trial: optuna.Trial) -> float:
        trial_dir = SAVE_PLOTS_ROOT / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        epochs = trial.suggest_int("epochs", 40, 120)

        # Pseudo-labeling block (no duplicate 'focal_alpha')
        if enable_pseudo_labeling:
            pl_hi = max(8, min(epochs // 2, epochs - 5))
            h = dict(
                PL_ENABLE=False,
                PL_START_EPOCH=pl_hi,
                PL_UPDATE_EVERY=trial.suggest_int("PL_UPDATE_EVERY", 1, 8),
                K_EACH=trial.suggest_int("k_each", 50, 400),
                PL_WEIGHT=trial.suggest_float("pl_weight", 0.5, 2.0),
            )

            # Consistency regularization — make sure lambda > 0 so KL contributes
            h.update(dict(
                CONS_ENABLE=True,
                CONS_RAMP_EPOCHS=trial.suggest_int("cons_ramp_epochs", 0, max(0, epochs // 3)),
                CONS_LAMBDA=trial.suggest_float("cons_lambda", 0.05, 1.0),
                CONS_T=trial.suggest_float("cons_T", 0.5, 2.0),
                CONS_ON=trial.suggest_categorical("cons_on", ["target_all", "target_unlabeled"]),
                CONS_CONF_GAMMA=trial.suggest_float("cons_conf_gamma", 1.0, 3.0),
                AUG_NOISE_STD=trial.suggest_float("aug_noise_std", 0.005, 0.05),
                AUG_GAIN_JITTER=trial.suggest_float("aug_gain_jitter", 0.0, 0.15),
                # optional toggles if you want to gate addition (your train_ae currently adds unconditionally)
                CONS_ADD_TO_TOTAL=True,
                CONS_IN_VAL_TOTAL=True,
            ))
        else:
            h = dict(PL_ENABLE=False)

        # Common hyperparams
        h.update(dict(
            AE_EPOCHS=epochs,
            AE_BATCH=trial.suggest_categorical("AE_BATCH", [16,32, 64, 128]),
            GRL_LAMBDA_MAX=trial.suggest_float("grl_lambda_max", 0.2, 2.0, step=0.1),
            FOCAL_GAMMA=trial.suggest_float("focal_gamma", 2.0, 3.5),
          
            PL_DOMAIN_ID=2,
            ENC_CONST_COEF=trial.suggest_categorical("enc_const", [0.5, 1.0, 1.5, 2.0]),
            DEC_CONST_COEF=trial.suggest_categorical("dec_const", [0.5, 1.0, 1.5, 2.0]),
            LATENT_DIM=trial.suggest_categorical("latent_dim", [8, 16, 32, 64, 128]),
            AE_PATIENCE=trial.suggest_int("ae_patience", 5, 12),
            COSINE_ALPHA=trial.suggest_float("cosine_alpha", 0.02, 0.30),
            LR_WARMUP_EPOCHS=trial.suggest_categorical("lr_warmup_epochs", [0, 2, 4, 6]),
            BASE_LR=trial.suggest_float("base_lr", 1e-5, 5e-4, log=True),
            CLIPNORM=trial.suggest_categorical("clipnorm", [1.0, 3.0, 5.0]),
            GRL_PEAK_FRAC=trial.suggest_float("grl_peak_frac", 0.2, 1.0),
            DOM_LABEL_SMOOTH=trial.suggest_categorical("dom_label_smoothing", [0.0, 0.05, 0.1, 0.15]),
            DOM_WEIGHT_TARGET= trial.suggest_float("dom_weight_target", 0.8, 2.5),
            ENT_TEMP=trial.suggest_categorical("ent_temp", [0.5, 1.0, 1.5, 2.0]),
            LW_REC_START=3.0,#trial.suggest_categorical("lw_rec_start", [2.0, 3.0, 4.0]),
            LW_DOM_START=1.0,#trial.suggest_categorical("lw_dom_start", [0.5, 1.0, 1.5]),
            LW_LOS_START=1.0,#trial.suggest_categorical("lw_los_start", [0.5, 1.0, 1.5]),
            LW_REC_END=0.5,#trial.suggest_categorical("lw_rec_end", [0.3, 0.5, 1.0]),
            LW_DOM_END=2.5,#trial.suggest_categorical("lw_dom_end", [1.5, 2.0, 2.5, 3.0]),
            LW_LOS_END=3.0,#trial.suggest_categorical("lw_los_end", [2.0, 2.5, 3.0, 4.0]),
            UNFREEZE_STEP=10,#trial.suggest_categorical("unfreeze_step_epochs", [5, 10, 15]),
            METRIC_THRESHOLD=.5,#trial.suggest_float("metric_threshold", 0.2, 0.6),
            SEED=SEED,
            save_plots=trial_dir,
            TRAIN_SIZE=TRAIN_SIZE,
            TRAIN1_NAME=DATASET_NAMES[0],
            TRAIN2_NAME=DATASET_NAMES[1],
            TEST_NAME=DATASET_NAMES[2],
            Plots_enabled=True,
            ADAPTION_WITH_LABEL=ADAPTION_WITH_LABEL,
            MIXED_PRECISION=USE_MIXED_PRECISION,
            MICRO_BATCH=32,
            w  = trial.suggest_float("w_val_vs_margin", 0.4, 0.8) ,        # weight on val AUROC
            mu = trial.suggest_float("entropy_penalty_mu", 0.0, 0.3)      # tiny penalty strength

        ))

        ae_model, _, val_acc, f1_valid, cm_valid, training_time, f1ent,val_auroc, optuna_score = train_ae(
            balanced_dsets, CIRS, RNG, Domains, Weights, Labels, h=h, trial=trial
        )

        ae_model.save(trial_dir / "ae_model.keras")

       
        del ae_model
        tf.keras.backend.clear_session()
        gc.collect()
        

        

        tf.keras.backend.clear_session()
        return float(val_acc)
    
    # ───────────────────────── Run the study ───────────────────────────
    def run_optuna(n_trials: int = 100, direction: str = "maximize") -> None:
        """Create a study, run optimisation, aggregate configs."""
        study = optuna.create_study(
            direction=direction,
            #sampler=optuna.samplers.TPESampler(seed=SEED)
           # pruner=optuna.pruners.MedianPruner(n_warmup_steps=4),
           sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=20),
           pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

        )

        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        # ─ summary ─
        print("\n═════════ Best trial ═════════")
        print("Trial # :", study.best_trial.number)
        print("Score   :", study.best_value)
        print("Params  :", study.best_params)

        # ─ aggregate every config dict ─
        records = {
            f"trial_{t.number:03d}": t.user_attrs[CONFIG_ATTR]
            for t in study.trials
            if t.state == TrialState.COMPLETE and CONFIG_ATTR in t.user_attrs
        }

        if records:
            (pd.DataFrame(records)
            .T.reset_index()
            .rename(columns={"index": "trial_name"})
            .to_excel(SAVE_PLOTS_ROOT/WHOLE_RES_XLSX, index=False))
            print(f"\n📝  Saved {len(records)} configs → {WHOLE_RES_XLSX}")
        else:
            print("\n⚠️   No completed trials produced configs – skipped aggregation.")

        # optional: full Optuna history
        study.trials_dataframe().to_csv(SAVE_PLOTS_ROOT/"optuna_results.csv", index=False)
        print("📊  Full trial history → optuna_results.csv")
        combine_trial_reports(SAVE_PLOTS_ROOT)


    # ───────────────────────── CLI entry‑point ─────────────────────────
    import argparse
    if __name__ == "__main__":
    

        parser = argparse.ArgumentParser(
            description="Optuna tuning wrapper for `main`."
        )

        # run exactly 100 trials unless the user passes a different value
        parser.add_argument(
            "--n-trials", type=int, default=60, help="Number of trials."
        )
        parser.add_argument(
            "--direction",
            choices=["maximize", "minimize"],
            default="maximize",               # keep maximization
            help="Optimisation direction."
        )

        #args = parser.parse_args()
        args = parser.parse_args(args=[])
        run_optuna(n_trials=args.n_trials, direction=args.direction)
