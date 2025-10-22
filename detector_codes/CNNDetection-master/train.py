import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # Setup logging directories
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize tensorboard writers
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))
    val_writer = SummaryWriter(os.path.join(log_dir, "val"))

    # Initialize training log
    log_file = os.path.join(log_dir, 'training_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("# Training Log for experiment: {}\n".format(opt.name))
            f.write(
                "# Started at: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S'))
            )
            f.write(
                "# Format: [timestamp] epoch step || train_loss train_acc lr || val_loss val_acc val_ap real_acc fake_acc\n"
            )
            f.write("#" + "=" * 100 + "\n\n")

    model = Trainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # Track training accuracy
            with torch.no_grad():
                output = model.model(model.input)
                pred = torch.sigmoid(output) > 0.5
                correct = (pred.squeeze() == model.label).sum().item()
                total_correct += correct
                total_samples += model.label.size(0)
                total_loss += model.loss.item()

            if model.total_steps % opt.loss_freq == 0:
                train_acc = total_correct / total_samples
                avg_loss = total_loss / (i + 1)
                print(
                    f"Train loss: {avg_loss:.4f}, acc: {train_acc:.4f} at step: {model.total_steps}"
                )
                train_writer.add_scalar('loss', avg_loss, model.total_steps)
                train_writer.add_scalar(
                    'accuracy', train_acc, model.total_steps
                )
                train_writer.add_scalar(
                    'learning_rate',
                    model.optimizer.param_groups[0]['lr'],
                    model.total_steps,
                )

            if model.total_steps % opt.save_latest_freq == 0:
                print(
                    'saving the latest model %s (epoch %d, model.total_steps %d)'
                    % (opt.name, epoch, model.total_steps)
                )
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(
                'saving the model at the end of epoch %d, iters %d'
                % (epoch, model.total_steps)
            )
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap, r_acc, f_acc, val_loss = validate(model.model, val_opt)

        # Log validation metrics
        val_writer.add_scalar('loss', val_loss, model.total_steps)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        val_writer.add_scalar('real_accuracy', r_acc, model.total_steps)
        val_writer.add_scalar('fake_accuracy', f_acc, model.total_steps)

        # Log epoch summary
        epoch_stats = {
            'train_loss': total_loss / len(data_loader),
            'train_acc': total_correct / total_samples,
            'train_lr': model.optimizer.param_groups[0]['lr'],
            'val_loss': val_loss,
            'val_acc': acc,
            'val_ap': ap,
            'val_real_acc': r_acc,
            'val_fake_acc': f_acc,
        }

        print(f"\nEpoch {epoch} Summary:")
        print(
            f"Training: loss={epoch_stats['train_loss']:.4f}, acc={epoch_stats['train_acc']:.4f}, lr={epoch_stats['train_lr']:.6f}"
        )
        print(
            f"Validation: loss={epoch_stats['val_loss']:.4f}, acc={epoch_stats['val_acc']:.4f}, ap={epoch_stats['val_ap']:.4f}"
        )
        print(
            f"Val Class Acc: real={epoch_stats['val_real_acc']:.4f}, fake={epoch_stats['val_fake_acc']:.4f}\n"
        )

        # Write metrics in an easily readable format
        try:
            log_file = os.path.join(
                opt.checkpoints_dir, opt.name, 'training_log.txt'
            )
            with open(log_file, 'a') as f:
                # Format: [timestamp] epoch step || train_loss train_acc lr || val_loss val_acc val_ap real_acc fake_acc
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                metrics_line = (
                    f"[{timestamp}] "
                    f"E{epoch:03d} S{model.total_steps:06d} || "
                    f"Loss: {epoch_stats['train_loss']:.4f} "
                    f"Acc: {epoch_stats['train_acc']:.4f} "
                    f"LR: {epoch_stats['train_lr']:.2e} || "
                    f"ValLoss: {epoch_stats['val_loss']:.4f} "
                    f"ValAcc: {epoch_stats['val_acc']:.4f} "
                    f"AP: {epoch_stats['val_ap']:.4f} "
                    f"(R: {epoch_stats['val_real_acc']:.4f} "
                    f"F: {epoch_stats['val_fake_acc']:.4f})"
                )
                f.write(metrics_line + '\n')
        except Exception as e:
            print(f"Failed to write metrics to log: {e}")

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                )
            else:
                print("Early stopping.")
                break
        model.train()
