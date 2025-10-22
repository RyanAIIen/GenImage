# train_f3net.py  (calls the F3Net model defined in f3net.py)
import os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from f3net import F3Net

# ---- paths and settings ----
DATA_ROOT = "dataset"  # symlinked view: dataset/train/{real,fake}, dataset/val/{real,fake}
CKPT_DIR  = "./runs"
BS, EPOCHS, LR, NUM_WORKERS = 64, 50, 1e-3, 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = F3Net(num_classes=2, img_width=299, img_height=299).to(device)

# transforms to match Xception input
tfm = transforms.Compose([
    transforms.Resize(333), transforms.CenterCrop(299),
    transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)
])

train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT,"train"), tfm)
val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT,"val"), tfm)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best = 0.0
os.makedirs(CKPT_DIR, exist_ok=True)

def run_epoch(loader, train=True):
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            _, logits = model(x)      # F3Net returns (features, logits)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        tot += y.size(0); correct += (logits.argmax(1)==y).sum().item(); loss_sum += loss.item()*y.size(0)
    return loss_sum/tot, correct/tot

for e in range(1, EPOCHS+1):
    t0=time.time()
    trL,trA = run_epoch(train_loader, True)
    vaL,vaA = run_epoch(val_loader,   False)
    print(f"[{e:03d}/{EPOCHS}] train {trL:.4f}/{trA:.4f} | val {vaL:.4f}/{vaA:.4f} | {time.time()-t0:.1f}s")
    if vaA > best:
        best = vaA
        torch.save(model.state_dict(), os.path.join(CKPT_DIR,"f3net_best.pth"))
        print(f"  -> saved best (val_acc={best:.4f})")

