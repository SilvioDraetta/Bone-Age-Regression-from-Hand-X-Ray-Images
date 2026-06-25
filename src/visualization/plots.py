#DEFINIRE FUNZIONE DI VISUALIZZAZIONE LOSS
import matplotlib.pyplot as plt
def plot_loss(history):
    """
    Plot training and validation loss as a function of training epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by model.fit(). It must contain the
        keys "loss" and "val_loss" in the history.history
        dictionary.

    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.show()