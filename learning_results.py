import pickle

import pandas as pd
from matplotlib import pyplot as plt

def compare_models_losses():
    models = [
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_20250802_0816.pickle", "name": "Original model"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_backbone_ca_20250803_0209.pickle", "name": "Backbone CA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_backbone_ca_head_ca_20250804_0718.pickle", "name": "Backbone CA + Head CA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_backbone_ca_head_eca_20250804_1246.pickle", "name": "Backbone CA + Head ECA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_backbone_ca_head_se_20250804_1808.pickle", "name": "Backbone CA + Head SE"},
    ]
    plt.figure(figsize=(len(models) * 4, 6))
    for idx, m in enumerate(models):

        with open(m["path"], "rb") as f:
            training_stats = pickle.load(f)
            train_df = pd.DataFrame(training_stats['train'])
            val_df = pd.DataFrame(training_stats['val'])
            epochs = list(range(1, len(train_df) + 1))
            plt.subplot(1, len(models), idx+1)
            plt.plot(epochs, train_df['total_loss'], label="Train")
            plt.plot(epochs, val_df['total_loss'], label="Val")
            plt.axhline(y=300, color='red', linestyle='--')
            plt.axhline(y=400, color='red', linestyle='--')
            plt.axhline(y=500, color='red', linestyle='--')
            plt.title(m["name"])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ax = plt.gca()
            ax.set_ylim([200, 1800])
            plt.legend()

    plt.tight_layout()
    plt.show()


def print_model_loss():
    file_loc = "/Users/sergebishyr/PhD/models/ball_detection/100e/ts_ssd_20250802_0816.pickle"
    with open(file_loc, "rb") as f:
        training_stats = pickle.load(f)
    train_df = pd.DataFrame(training_stats['train'])
    val_df = pd.DataFrame(training_stats['val'])
    epochs = list(range(1, len(train_df) + 1))
    plt.figure(figsize=(14, 6))
    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_df['total_loss'], label="Train")
    plt.plot(epochs, val_df['total_loss'], label="Val")
    plt.axhline(y=300, color='red', linestyle='--')
    plt.axhline(y=500, color='red', linestyle='--')
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Localization Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_df['loc_loss'], label="Train")
    plt.plot(epochs, val_df['loc_loss'], label="Val")
    plt.title("Localization Loss (bbox_regression)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Classification Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_df['cls_loss'], label="Train")
    plt.plot(epochs, val_df['cls_loss'], label="Val")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # print_model_loss()
    compare_models_losses()

