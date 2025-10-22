from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder
import torch
import torchvision.datasets as datasets

def safe_loader(path):
  try:
    with open(path,"rb") as f:
      img = Image.open(f)
      return img.convert("RGB")
  except (UnidentifiedImageError, OSError) as e:
    print("Warning skipping corrupter image")
    return None

class SafeImageFolder(ImageFolder):
  def __init__(self,root,transform = None, target_transform=None):
    super().__init__(root,transform = transform, target_transform = target_transform, loader= safe_loader)


  def __getitem__(self,index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if sample is None:
      return None
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return sample, target  

def collate_fn(batch):
  batch = [x for x in batch if x is not None and x[0] is not None and x[1] is not None]

  if len(batch) == 0:
    print("[WARNING] Skipping empty batch after filtering corrupted images")
    return torch.empty(0), torch.empty(0)
  return torch.utils.data.dataloader.default_collate(batch)

import main
if __name__ == "__main__":
  main.__name__== "__main__"
  exec(open("main.py").read())


