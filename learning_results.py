import pickle

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    file_loc = "/Users/sergebishyr/PhD/models/ball_detection/ssd_attention_crop_300_7aa39cdbadd65be59321ec520834dcf77e680497/training_stats_ssd_20250713_1652.pickle"
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

