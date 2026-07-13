import queue
import threading
import torch
from rich.console import Console

from vorotracing.datasets.datasets import Dataset

CONSOLE = Console(width=120)


class RayBatcher(threading.Thread):
    def __init__(self, dataset: Dataset, rays_per_batch: int, device="cuda", patch_size: int = 0):
        """
        High-performance ray provider for NeRF.

        This class creates a background thread that prefetches batches of rays from the dataset and puts them in a queue.
        A seperate CUDA stream is used to perform the copy and ray math operations on the GPU. This allows these operations
        to be hidden in the gaps between other CUDA kernels.

        Args:
            dataset: Dataset object
            rays_per_batch: Rays per training step
        """
        super().__init__()

        assert dataset.split == "train", "RayBatcher is only for training"

        self.dataset = dataset
        self.dataset.pin_memory()

        self.rays_per_batch = rays_per_batch
        self.device = device
        # patch_size > 0 -> sample warp-coherent PxP patches instead of random rays
        self.patch_size = patch_size

        # 1. Create a dedicated stream for background work (copy and ray math)
        self.stream = torch.cuda.Stream()

        # Prefetch buffer (Queue of 3 batches)
        self._queue = queue.Queue(maxsize=3)
        self.daemon = True
        self.start()

    def _prepare_batch(self):
        """Heavy CPU operations happen here in the background thread."""
        if self.patch_size and self.patch_size > 0:
            batch = self.dataset.sample_patch_batch(self.rays_per_batch, self.patch_size)
        else:
            batch = self.dataset.sample_batch(self.rays_per_batch)
        return batch

    def run(self):
        """Continuously populates the queue. Overloades threading.Thread.run() running in a separate thread."""
        while True:
            with torch.cuda.stream(self.stream):
                batch = self._prepare_batch()
                # This blocks if the queue has 3 batches, waiting for the GPU to finish one.
                self._queue.put(batch)

    def next_batch(self):
        """Called by the training loop to get the next batch."""
        batch = self._queue.get()

        # 4. CRITICAL: Tell the main (default) stream to wait for the
        # background stream to finish the copy/math for THIS specific batch.
        torch.cuda.current_stream().wait_stream(self.stream)

        return batch
        # Copy to GPU and preprocess
        # return self.dataset.preprocess_batch(batch)
