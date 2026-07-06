import time

def time_dataloader(dataloader, n_batches=1):
    """
    Measure the time required to fetch batches from a PyTorch DataLoader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The DataLoader to benchmark.

    n_batches : int, optional (default=1)
        Number of consecutive batches to fetch. Useful to average
        performance over multiple iterations.

    Returns
    -------
    float
        Total elapsed time in seconds required to fetch `n_batches`
        batches from the DataLoader.
    """
    t0 = time.time()

    iterator = iter(dataloader)
    for _ in range(n_batches):
        next(iterator)

    t1 = time.time()
    print("=== TEST DATALOADER ===")
    print("DataLoader time:", t1 - t0)


import torch

def time_forward(model, dataloader, device):
    """
    Measure the forward-pass time of a PyTorch model using a single batch
    from the provided DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark. It will be temporarily set to eval mode.

    dataloader : torch.utils.data.DataLoader
        DataLoader from which the first batch will be extracted.

    device : torch.device
        Device on which the model and tensors are allocated.

    Returns
    -------
    float
        Elapsed time in seconds for a single forward pass.
    """
    batch = next(iter(dataloader))

    image = batch["image"].to(device)
    male = batch["male"].float().unsqueeze(1).to(device)

    model.eval()

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        _ = model(image, male)

    torch.cuda.synchronize()
    t1 = time.time()

    print("=== TEST FORWARD ===")
    print("Forward time:", t1 - t0)

