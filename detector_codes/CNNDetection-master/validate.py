import torch
import numpy as np
import os
import time
from networks.resnet import resnet50
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
)
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"\nValidating on {len(data_loader)} batches...")
    with torch.no_grad():
        y_true, y_pred = [], []
        for i, (img, label) in enumerate(data_loader):
            if i % 10 == 0:  # Show progress every 10 batches
                print(f"Processing batch {i}/{len(data_loader)} ({(i/len(data_loader)*100):.1f}%)", end='\r')

            in_tens = img.cuda()
            label = label.cuda()
            output = model(in_tens)
            loss = criterion(output.flatten(), label.float())
            total_loss += loss.item()

            y_pred.extend(output.sigmoid().flatten().cpu().tolist())
            y_true.extend(label.flatten().cpu().tolist())
        print("\nValidation complete!")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    avg_loss = total_loss / len(data_loader)

    return acc, ap, r_acc, f_acc, avg_loss


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, avg_loss = validate(model, opt)

    # Prepare results message
    results_message = f"""----------------- Validation Results ---------------
Model path: {opt.model_path}
Dataset: {opt.dataroot}
Split: {opt.train_split}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Metrics:
    Accuracy: {acc:.4f}
    Average Precision: {avg_precision:.4f}
    Real Images Accuracy: {r_acc:.4f}
    Fake Images Accuracy: {f_acc:.4f}
    Average Loss: {avg_loss:.4f}
-----------------  End  ---------------"""

    # Print to console
    print(results_message)

    # Save to file in the checkpoints directory with dataset name and timestamp
    checkpoint_dir = os.path.dirname(opt.model_path)
    timestamp = time.strftime('%Y%m%dT%H%M%S')  # ISO 8601 format

    # Extract dataset name from path (parent of val directory)
    dataset_path = os.path.dirname(os.path.normpath(opt.dataroot))  # Get parent of val directory
    dataset_name = os.path.basename(dataset_path)  # Get the dataset name

    # Extract model version from path (e.g., 'latest' from model_epoch_latest.pth)
    model_file = os.path.basename(opt.model_path)
    model_version = model_file.replace('model_epoch_', '').replace('.pth', '')

    results_file = os.path.join(checkpoint_dir, f'validation_results_{timestamp}_{model_version}_{dataset_name}.txt')    # Write results to new file
    with open(results_file, 'w') as f:
        f.write(results_message + "\n")
