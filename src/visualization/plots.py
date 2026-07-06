import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(path="name_excel.xlsx"):
    df = pd.read_excel(path)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()