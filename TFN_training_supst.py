#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os
import tqdm

from models.model_selector import *
from models.TransFlowNet import *
import models.TransFlowNet_sups as TFN_s
import models.TransFlowNet_supst_d4 as TFN_st_4
import models.flow as flow


def main():
    parser = argparse.ArgumentParser("Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./dataset/zheyi/",
        type=str,
        help="training data path; subfolders: ges, labels",
    )
    parser.add_argument(
        "--work_dir", default="workdir/zheyi", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="unet", help="select mode: unet, unetr, swinunetr， swinunetr_dfc_v3"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=50, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--model_path", type=str, default="unet_sups")

    args = parser.parse_args()

    from monai.utils import GridSampleMode

    join = os.path.join

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    import monai
    from monai.data import decollate_batch, PILReader
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activations,
        AddChanneld,
        AsDiscrete,
        Compose,
        LoadImaged,
        SpatialPadd,
        RandSpatialCropd,
        RandRotate90d,
        ScaleIntensityd,
        RandAxisFlipd,
        RandZoomd,
        RandGaussianNoised,
        RandAdjustContrastd,
        RandGaussianSmoothd,
        RandHistogramShiftd,
        EnsureTyped,
        EnsureType, EnsureChannelFirstd,
        Rand2DElasticd, GaussianSmooth
    )
    from monai.visualize import plot_2d_or_3d_image
    from datetime import datetime
    import shutil

    print("Successfully imported all requirements!")

    monai.config.print_config()

    # %% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_path)
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.1
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    # %% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8
            ),
            AddChanneld(keys=["img", "label"], allow_missing_keys=True),  # label: (1, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label

            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2), sigma_y=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["img", "label"], allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    # % define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    # %% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    post_pred = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TFN_st_4.TransFlowNet(args.model_name.lower(), device, args, in_channels=1)

    # model = model_factory(args.model_name.lower(), device, args, in_channels=1)

    # loss_function = monai.losses.DiceCELoss(softmax=True).to(device)
    loss_function = monai.losses.DiceCELoss(sigmoid=True)
    sloss_function = nn.MSELoss()

    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    # smooth_transformer = GaussianSmooth(sigma=1)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        train_bar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for step, batch_data in train_bar:
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            # sup s_normal xy_normal
            s = flow.get_s(labels)
            xy = flow.get_t(labels.cpu()).to(device).float()

            _, _, outputs, s_normal, xy_normal = model(inputs)  # s_normal (b, 1, 1, 1)

            loss = loss_function(outputs, labels) + 0.1 * sloss_function(s_normal, s) + 0.01 * sloss_function(xy_normal, xy)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size

            train_bar.set_postfix_str(f"train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        if epoch >= 10 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_hidden_feature = None
                val_forward = None

                for step, val_data in enumerate(val_loader, 1):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)

                    val_labels_onehot = val_labels
                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = args.batch_size

                    val_forward, val_hidden_feature, val_outputs, s_normal, xy_normal = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_hidden_feature = [post_pred(i) for i in decollate_batch(val_hidden_feature)]
                    val_forward = [i for i in decollate_batch(val_forward)]

                    val_labels_onehot = [
                        post_gt(i) for i in decollate_batch(val_labels_onehot)
                    ]

                    dice = dice_metric(y_pred=val_outputs, y=val_labels_onehot)

                    print(os.path.basename(
                        val_data["img_meta_dict"]["filename_or_obj"][0]
                    ), dice)

                # aggregate the final mean f1 score and dice result
                dice_metric_ = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                if dice_metric_ > best_metric:
                    best_metric = dice_metric_
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, dice_metric_, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalars("val_metrics", {"dice": dice_metric_}, epoch + 1)

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output", max_channels=3)
                plot_2d_or_3d_image(val_hidden_feature, epoch, writer, index=0, tag="hidden")
                plot_2d_or_3d_image(val_forward, epoch, writer, index=0, tag="forward")
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}"
                )
                break

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()