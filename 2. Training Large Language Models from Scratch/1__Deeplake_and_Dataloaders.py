from dotenv import load_dotenv
load_dotenv()
import os

import deeplake
from torch.utils.data import DataLoader, Dataset

# create dataset on DeepLake
username = os.environ["ACTIVELOOP_ORG_ID"]
dataset_name = "test_dataset"
ds = deeplake.dataset(f"hub://{username}/{dataset_name}")

# create column text
ds.create_tensor('text', htype="text")

# add some texts to the dataset
texts = [f"text {i}" for i in range(1, 11)]
for text in texts:
    ds.append({"text": text})

# create PyTorch data loader
batch_size = 3
train_loader = ds.dataloader()\
    .batch(batch_size)\
    .shuffle()\
    .pytorch()

# loop over the elements
for i, batch in enumerate(train_loader):
    print(f"Batch {i}")
    samples = batch.get("text")
    for j, sample in enumerate(samples):
        print(f"Sample {j}: {sample}")
    print()
    pass

class DeepLakePyTorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        texts = self.ds.text[idx].text().astype(str)
        return { "text": texts }

# create PyTorch dataset
ds_pt = DeepLakePyTorchDataset(ds)

# create PyTorch data loader from PyTorch dataset
dataloader_pytorch = DataLoader(ds_pt, batch_size=3, shuffle=True)

# loop over the elements
for i, batch in enumerate(dataloader_pytorch):
    print(f"Batch {i}")
    samples = batch.get("text")
    for j, sample in enumerate(samples):
        print(f"Sample {j}: {sample}")
    print()
    pass