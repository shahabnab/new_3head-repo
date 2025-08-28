from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap
import os
import pandas as pd
from scipy.stats import entropy
from tensorflow.keras import Model
import tensorflow as tf


def plot_confusion_matrix(classifier,save_path, CIRS,Y,Label_name,thr,title="Confusion Matrix"):
    # 1) get predicted probabilities (shape: (n_samples, 1) or similar)
    X      = CIRS[Label_name][..., None]    # shape: (N, seq_len, 1)
    y_true=Y[Label_name]



    probs = classifier.predict(X)

    # 2) threshold at 0.5 to get binary predictions
    predictions = np.where(probs > thr, 1, 0).flatten()

    # 3) print classification report
    print(classification_report(y_true, predictions, target_names=["Class 0", "Class 1"]))

    # 4) compute confusion matrix
    cm = confusion_matrix(y_true, predictions)
    print("Confusion Matrix:\n")
    print(cm,"\n")
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy: {accuracy:.2f}")

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, predictions)
    print(f"Accuracy (via sklearn): {acc:.2f}")


    # 5) plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["LOS", "NLOS"],
        yticklabels=["LOS", "NLOS"]
    )
    plt.title(title)
    plt.xlabel(f"Predicted\nAccuracy: {accuracy:.2f}")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{title} {Label_name}.png")
 
    plt.close()



def plot_latent_umap(encoder_model, CIRS, Domains,save_plots, title="UMAP of Latent Space"):
    X = np.concatenate([CIRS[k] for k in ("TRAIN1", "TRAIN2", "ADAPTION")])[..., None]
    d = np.concatenate([Domains[k] for k in ("TRAIN1", "TRAIN2", "ADAPTION")])

    latent_vectors = encoder_model.predict(X)
    scaled_latents = StandardScaler().fit_transform(latent_vectors)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_latents)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=d, palette="Set2", s=30)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Domain")
    plt.grid(True)
    
    plt.savefig(os.path.join(save_plots, "latent_umap_latent_vector.png"))
    plt.close()




def plot_latent_umap2(encoder_model, CIRS, Domains,save_plots, title="UMAP of Latent Space"):
    X = np.concatenate([CIRS[k] for k in ("TRAIN1", "TRAIN2", "ADAPTION")])
    d = np.concatenate([Domains[k] for k in ("TRAIN1", "TRAIN2", "ADAPTION")])

    # Flatten each sample if X is 2D (samples, timesteps), else reshape appropriately
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X

    scaled_input = StandardScaler().fit_transform(X_flat)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_input)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=d, palette="Set2", s=30)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Domain")
    plt.grid(True)
    plt.savefig(os.path.join(save_plots, "latent_umap_input.png"))
    plt.close()   




def signal_entropy(signal):
    """Compute normalized Shannon entropy of a signal."""
    prob_dist, _ = np.histogram(signal, bins=32, density=True)
    prob_dist = prob_dist[prob_dist > 0]  # avoid log(0)
    return entropy(prob_dist) / np.log(len(prob_dist))

def compute_complexity_metrics(CIRS):
    metrics = {
        "domain": [],
        "variance": [],
        "peak_to_mean": [],
        "entropy": [],
    }

    for domain_name, signals in CIRS.items():
        for sig in signals:
            metrics["domain"].append(domain_name)
            metrics["variance"].append(np.var(sig))
            metrics["peak_to_mean"].append(np.max(sig) / (np.mean(sig) + 1e-6))
            metrics["entropy"].append(signal_entropy(sig))

    return metrics

def plot_complexity_metrics(CIRS):
    """Compute and plot complexity metrics for each domain in CIRS."""
    metrics = compute_complexity_metrics(CIRS)
    df_metrics = pd.DataFrame(metrics)
    # Boxplots for comparison
    plt.figure(figsize=(14, 4))
    for i, metric in enumerate(["variance", "peak_to_mean", "entropy"]):
        plt.subplot(1, 3, i+1)
        sns.boxplot(data=df_metrics, x="domain", y=metric)
        plt.title(metric.capitalize())
    plt.tight_layout()
    plt.savefig("complexity_metrics.png")
    plt.close() 


def plot_encoded_signals(ae,decoder,encoder, CIRS, labels, save_plots):
    
    ENC= tf.keras.Model(
    inputs  = ae.input,                               # (None, seq_len, 1)
    outputs = ae.get_layer("latent_vector").output,   # last encoder node
    name    = "encoder_trunk"
    )

    train1_los=CIRS["TRAIN1"][labels["TRAIN1"]==0][..., None]
    train2_los=CIRS["TRAIN2"][labels["TRAIN2"]==0][..., None]
    adaption_los=CIRS["ADAPTION"][labels["ADAPTION"]==0][..., None]
    test_los=CIRS["TEST"][labels["TEST"]==0][..., None]


    train1_nlos=CIRS["TRAIN1"][labels["TRAIN1"]==1][..., None]
    train2_nlos=CIRS["TRAIN2"][labels["TRAIN2"]==1][..., None]
    adaption_nlos=CIRS["ADAPTION"][labels["ADAPTION"]==1][..., None]
    test_nlos=CIRS["TEST"][labels["TEST"]==1][..., None]


    # %%
    plt.figure(figsize=(12, 6))
    plt.plot(train1_los[0], label='Train1 LOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(train1_los[0], axis=0)))[0], label='Decoded Train1 LOS')
    plt.title('Train1 LOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "train1_los_decoded.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(train1_nlos[0], label='Train1 NLOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(train1_nlos[0], axis=0)))[0], label='Decoded Train1 NLOS')
    plt.title('Train1 NLOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "train1_nlos_decoded.png"))
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.plot(train1_los[0], label='Train2 LOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(train2_los[0], axis=0)))[0], label='Decoded Train2 LOS')
    plt.title('Train2 LOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "train2_los_decoded.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(train2_nlos[0], label='Train2 NLOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(train2_nlos[0], axis=0)))[0], label='Decoded Train2 NLOS')
    plt.title('Train2 NLOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "train2_nlos_decoded.png"))
    plt.close()



    plt.figure(figsize=(12, 6))
    plt.plot(adaption_los[0], label='Adaption LOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(adaption_los[0], axis=0)))[0], label='Decoded Adaption LOS')
    plt.title('Adaption LOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "adaption_los_decoded.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(adaption_nlos[0], label='Adaption NLOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(adaption_nlos[0], axis=0)))[0], label='Decoded Adaption NLOS')
    plt.title('Adaption NLOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "adaption_nlos_decoded.png"))
    plt.close()

    # %%
    plt.figure(figsize=(12, 6))
    plt.plot(test_nlos[0], label='TEST NLOS')

    plt.plot(decoder.predict(encoder.predict(np.expand_dims(test_nlos[0], axis=0)))[0], label='Decoded TEST NLOS')
    plt.title('TEST NLOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "test_nlos_decoded.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_los[0], label='TEST LOS')
    plt.plot(decoder.predict(encoder.predict(np.expand_dims(test_los[0], axis=0)))[0], label='Decoded TEST LOS')
    plt.title('TEST LOS Signal and its Decoded Versions')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(save_plots, "test_los_decoded.png"))
    plt.close()


def probe_ae(ae, CIRS,labels, save_plots):
    CIRS_test_LOS= CIRS["TEST"][labels["TEST"]==0][..., None]
    CIRS_test_NLOS= CIRS["TEST"][labels["TEST"]==1][..., None]
    for i in range(1, 2):
        print("CIRS_test_LOS shape:", CIRS_test_LOS[i].shape)
        print("CIRS_test_NLOS shape:", CIRS_test_NLOS[i].shape)
        # 1) Prepare LOS / NLOS with a proper batch dimension of 1
        x_los  = CIRS_test_LOS [i:i+1]   # shape: (1, 150, 1)
        x_nlos = CIRS_test_NLOS[i:i+1]   # shape: (1, 150, 1)

        # 2) Layers to probe
        """ probe_layer_names = [
            "latent_vector",      # 16-dim bottleneck
            "dense",              # 64-unit Dense (replace latent_cls)
            "multiply",           # Gated output before los_logits (replace latent_attention)
            "los_logits",         # final 1-dim sigmoid
        ] """
        probe_layer_names = [
            "latent_vector",
            "latent_cls",
            "attention_gated",
            "los_prob"
        ]

        # 3) Build a probe model
        probe_outputs = [ae.get_layer(name).output
                        for name in probe_layer_names]
        probe_model   = Model(inputs=ae.input,
                            outputs=probe_outputs)

        # 4) Run the probe
        los_vec, los_cls, los_attn, los_logits  = probe_model.predict(x_los)
        nlos_vec, nlos_cls, nlos_attn, nlos_logits = probe_model.predict(x_nlos)

        # 5) Squeeze away the batch axis
        los_vec    = los_vec.squeeze()     # → (16,)
        los_cls    = los_cls.squeeze()     # → (64,)
        los_attn   = los_attn.squeeze()    # → (64,)
        los_logits = float(los_logits)     # → scalar

        nlos_vec    = nlos_vec.squeeze()
        nlos_cls    = nlos_cls.squeeze()
        nlos_attn   = nlos_attn.squeeze()
        nlos_logits = float(nlos_logits)

        # 6) Plot everything
        fig, axes = plt.subplots(4, 1, figsize=(6, 12))

        # ── latent_vector (16 dims)
        axes[0].plot(los_vec,   marker='o', color='blue',   label='LOS')
        axes[0].plot(nlos_vec,  marker='x', linestyle='--', color='orange', label='NLOS')
        axes[0].set_title("latent_vector (16 dims)")
        axes[0].set_ylabel("Activation")
        axes[0].legend()

        # ── latent_cls (64 dims)
        axes[1].plot(los_cls,   marker='o', color='blue')
        axes[1].plot(nlos_cls,  marker='x', linestyle='--', color='orange')
        axes[1].set_title("latent_cls (64 dims)")
        axes[1].set_ylabel("Activation")

        # ── latent_attention (64 dims)
        axes[2].plot(los_attn,  marker='o', color='blue')
        axes[2].plot(nlos_attn, marker='x', linestyle='--', color='orange')
        axes[2].set_title("latent_attention (64 dims)")
        axes[2].set_ylabel("Activation")

        # ── los_logits (scalar probability)
        axes[3].bar([0], [los_logits],    width=0.4, color='blue',   label='LOS')
        axes[3].bar([1], [nlos_logits],   width=0.4, color='orange', label='NLOS')
        axes[3].set_xlim(-0.5, 1.5)
        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(['LOS', 'NLOS'])
        axes[3].set_title("los_logits (sigmoid output)")
        axes[3].set_ylabel("Probability")
        axes[3].legend()

        plt.xlabel("Latent‐dimension index (where applicable)")
        plt.tight_layout()
        plt.savefig(f"{save_plots}/probe_ae_{i}.png")
        plt.close()


def plot_all_histories(history, save_path):
    os.makedirs(save_path, exist_ok=True)

    # 1) Group keys by their metric name (suffix after train_/val_/test_)
    metric_groups = {}
    for key, values in history.items():
        if key.startswith("train_"):
            phase, base = "train", key[len("train_"):]
        elif key.startswith("val_"):
            phase, base = "val", key[len("val_"):]
        elif key.startswith("test_"):
            phase, base = "test", key[len("test_"):]
        else:
            # assume un-prefixed == training metric
            phase, base = "train", key

        metric_groups.setdefault(base, {})[phase] = values

    # 2) For each metric, plot train/val/test on the same figure
    for base, group in metric_groups.items():
        plt.figure()
        for phase in ("train", "val", "test"):
            if phase in group:
                plt.plot(group[phase], label=phase)
        plt.title(f"{base} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(base)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{base}.png"))
        plt.close()
import matplotlib.pyplot as plt

def plot_history(history,save_path):
    """
    Plot all metrics stored in the history dictionary in one figure.
    Each key will be plotted in its own subplot.
    """
    keys = list(history.keys())
    n_keys = len(keys)

    # figure size depends on number of keys
    n_cols = 3
    n_rows = (n_keys + n_cols - 1) // n_cols
    plt.figure(figsize=(6*n_cols, 4*n_rows))

    for i, key in enumerate(keys, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.plot(history[key], label=key, linewidth=1.8)
        plt.title(key, fontsize=10)
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"alltogether.png"))
    plt.close()


def plot_predicted_vs_real(y_pred, y_real, save_path):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(y_pred)), y_pred, color='blue', label='NLOS predicted')
        plt.scatter(np.arange(len(y_pred)), y_real, color='red', label='LOS real')
        plt.title('Predicted LOS/NLOS Probabilities')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Probability')
        plt.legend()
        plt.savefig(f"{save_path}/predicted_los_nlos_probabilities.png")
        plt.close()

def evaluate_model_performance(probe_model, CIRS, LosLabels, save_plots,TEXT_label):
        #TEXT_label="ADAPTION"  # or "TRAIN2", "ADAPTION", "TEST"
        CIRS_test_LOS= CIRS[TEXT_label][LosLabels[TEXT_label]==0][..., None]
        CIRS_test_NLOS= CIRS[TEXT_label][LosLabels[TEXT_label]==1][..., None]
        # get every test example’s predicted LOS-probability
        all_X = np.concatenate([CIRS_test_LOS, CIRS_test_NLOS], axis=0)
        all_y = np.concatenate([np.zeros(len(CIRS_test_LOS)),
                                np.ones(len(CIRS_test_NLOS))])
        all_probs = probe_model.predict(all_X)[-1].squeeze()
        plt.figure(figsize=(12, 6))
        plt.hist(all_probs[all_y==0], bins=30, alpha=0.6, label="LOS")
        plt.hist(all_probs[all_y==1], bins=30, alpha=0.6, label="NLOS")
        plt.xlabel("Predicted P(NLOS)")
        plt.ylabel("Count")
        plt.legend()
        plt.title(f"{TEXT_label} set probability distributions")
        plt.savefig(f"{save_plots}/{TEXT_label}_set_probabilities.png")

      
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(all_probs)), all_probs, alpha=0.6, label="LOS")
        plt.scatter(range(len(all_y)), all_y, alpha=0.6, label="NLOS")
        #plt.ylim([0.4, 0.7])
        plt.xlabel("Predicted P(NLOS)")
        plt.ylabel("Count")
        plt.legend()
        plt.title(f"{TEXT_label} set probability distributions")
        plt.savefig(f"{save_plots}/{TEXT_label}_set_probabilities_scatter.png")