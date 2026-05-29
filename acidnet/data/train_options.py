import argparse

from acidnet.paths import DATASET_ROOT, OUTPUT_ROOT


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


PROFILES = {
    "standard": {
        "batch_size": 8,
        "crop_size": 256,
        "lr": 1e-4,
        "edge_weight": 50,
        "perceptual_weight": 0.01,
        "dataset": "lol_v1",
    },
}


def _dataset_path(*parts):
    return str(DATASET_ROOT.joinpath(*parts))


def option(profile="standard"):
    defaults = PROFILES[profile]
    parser = argparse.ArgumentParser(description="ACIDNet training")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=profile)
    parser.add_argument("--batchSize", type=int, default=defaults["batch_size"], help="training batch size")
    parser.add_argument("--cropSize", type=int, default=defaults["crop_size"], help="image crop size")
    parser.add_argument("--nEpochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help="resume from this epoch")
    parser.add_argument("--snapshots", type=int, default=10, help="checkpoint interval")
    parser.add_argument("--lr", type=float, default=defaults["lr"], help="learning rate")
    parser.add_argument("--gpu_mode", type=_str2bool, default=True)
    parser.add_argument("--shuffle", type=_str2bool, default=True)
    parser.add_argument("--threads", type=int, default=16, help="dataloader workers")
    parser.add_argument("--dry_run", action="store_true", help="initialize data, model, optimizer, and losses without training")

    parser.add_argument("--cos_restart_cyclic", type=_str2bool, default=False)
    parser.add_argument("--cos_restart", type=_str2bool, default=True)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--start_warmup", type=_str2bool, default=True)

    parser.add_argument("--data_train_lol_blur", type=str, default=_dataset_path("LOL_blur/train"))
    parser.add_argument("--data_train_lol_v1", type=str, default=_dataset_path("LOLdataset/our485"))
    parser.add_argument("--data_train_lolv2_real", type=str, default=_dataset_path("LOLv2/Real_captured/Train"))
    parser.add_argument("--data_train_lolv2_syn", type=str, default=_dataset_path("LOLv2/Synthetic/Train"))
    parser.add_argument("--data_train_SID", type=str, default=_dataset_path("Sony_total_dark/train"))
    parser.add_argument("--data_train_SICE", type=str, default=_dataset_path("SICE/Dataset/train"))
    parser.add_argument("--data_train_fivek", type=str, default=_dataset_path("FiveK/train"))

    parser.add_argument("--data_val_lol_blur", type=str, default=_dataset_path("LOL_blur/eval/low_blur"))
    parser.add_argument("--data_val_lol_v1", type=str, default=_dataset_path("LOLdataset/eval15/low"))
    parser.add_argument("--data_val_lolv2_real", type=str, default=_dataset_path("LOLv2/Real_captured/Test/Low"))
    parser.add_argument("--data_val_lolv2_syn", type=str, default=_dataset_path("LOLv2/Synthetic/Test/Low"))
    parser.add_argument("--data_val_SID", type=str, default=_dataset_path("Sony_total_dark/eval/short"))
    parser.add_argument("--data_val_SICE_mix", type=str, default=_dataset_path("SICE/Dataset/eval/test"))
    parser.add_argument("--data_val_SICE_grad", type=str, default=_dataset_path("SICE/Dataset/eval/test"))
    parser.add_argument("--data_test_fivek", type=str, default=_dataset_path("FiveK/test/input"))

    parser.add_argument("--data_valgt_lol_blur", type=str, default=_dataset_path("LOL_blur/eval/high_sharp_scaled"))
    parser.add_argument("--data_valgt_lol_v1", type=str, default=_dataset_path("LOLdataset/eval15/high"))
    parser.add_argument("--data_valgt_lolv2_real", type=str, default=_dataset_path("LOLv2/Real_captured/Test/Normal"))
    parser.add_argument("--data_valgt_lolv2_syn", type=str, default=_dataset_path("LOLv2/Synthetic/Test/Normal"))
    parser.add_argument("--data_valgt_SID", type=str, default=_dataset_path("Sony_total_dark/eval/long"))
    parser.add_argument("--data_valgt_SICE_mix", type=str, default=_dataset_path("SICE/Dataset/eval/target"))
    parser.add_argument("--data_valgt_SICE_grad", type=str, default=_dataset_path("SICE/Dataset/eval/target"))
    parser.add_argument("--data_valgt_fivek", type=str, default=_dataset_path("FiveK/test/target"))

    parser.add_argument("--val_folder", default=str(OUTPUT_ROOT / "validation") + "/", help="validation output directory")

    parser.add_argument("--HVI_weight", type=float, default=1.0)
    parser.add_argument("--L1_weight", type=float, default=1.0)
    parser.add_argument("--D_weight", type=float, default=0.5)
    parser.add_argument("--E_weight", type=float, default=defaults["edge_weight"])
    parser.add_argument("--P_weight", type=float, default=defaults["perceptual_weight"])

    parser.add_argument("--gamma", type=_str2bool, default=False)
    parser.add_argument("--start_gamma", type=int, default=60)
    parser.add_argument("--end_gamma", type=int, default=120)
    parser.add_argument("--grad_detect", type=_str2bool, default=False)
    parser.add_argument("--grad_clip", type=_str2bool, default=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default=defaults["dataset"],
        choices=["lol_v1", "lolv2_real", "lolv2_syn", "lol_blur", "SID", "SICE_mix", "SICE_grad", "fivek"],
        help="dataset to train on",
    )
    return parser
