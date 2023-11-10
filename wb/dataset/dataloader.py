import torch


class NaiveDataLoader:
    def __init__(self, dataset, batch_size=64):
        self.index = 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.index = 0
        return self
    
    def collate_fn(self, batch):
        X, y = batch[0]
        x_batch = torch.zeros(len(batch), X.shape[0], X.shape[1], dtype=torch.float32 )
        y_batch = torch.zeros(len(batch), y.shape[0], y.shape[1], dtype=torch.float32 )
        for i, item in enumerate(batch):
            X, y = item
            x_batch[i, :, :] = torch.from_numpy(X)
            y_batch[i, :, :] = torch.from_numpy(y)

        return x_batch, y_batch

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        batch = [self.get() for _ in range(batch_size)]
        return self.collate_fn(batch)

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item