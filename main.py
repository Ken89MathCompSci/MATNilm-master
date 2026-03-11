import copy
import os
import utils
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from custom_types import Basic, TrainConfig
from modules import MATconv as MAT
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--hidden", type=int, default=32, help="encoder decoder hidden size")
    parser.add_argument("--logname", action="store", default='root', help="name for log")
    parser.add_argument("--subName", action="store", type=str, default='test', help="name of the directory of current run")
    parser.add_argument("--inputLength", type=int, default=864, help="input length for the model")
    parser.add_argument("--outputLength", type=int, default=864, help="output length for the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dataAug", action="store_true", help="data augmentation mode")
    parser.add_argument("--prob0", type=float, default=0.3, help="augment probability for Dishwasher")
    parser.add_argument("--prob1", type=float, default=0.6, help="weight")
    parser.add_argument("--prob2", type=float, default=0.3, help="weight")
    parser.add_argument("--prob3", type=float, default=0.3, help="weight")
    parser.add_argument("--epochs", type=int, default=80, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument("--no_early_stopping", action="store_true", help="disable early stopping and train for all epochs")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="All_best_onoff.ckpt", help="checkpoint file name to resume from")
    return parser.parse_args()


def plot_metrics(mae_history, sae_history, f1_history, modelDir):
    appliances = ['Dishwasher', 'Fridge', 'Microwave', 'Washing Machine']
    epochs = range(1, len(mae_history) + 1)
    mae_arr = np.array(mae_history)
    sae_arr = np.array(sae_history)
    f1_arr = np.array(f1_history)

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(appliances):
        axes[0].plot(epochs, mae_arr[:, i], label=name)
    axes[0].set_title('MAE per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (watts)')
    axes[0].legend()

    for i, name in enumerate(appliances):
        axes[1].plot(epochs, sae_arr[:, i], label=name)
    axes[1].set_title('SAE per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SAE (watts)')
    axes[1].legend()

    for i, name in enumerate(appliances):
        axes[2].plot(epochs, f1_arr[:, i], label=name)
    axes[2].set_title('F1 Score per Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, 'metrics.png'), dpi=150)
    plt.show()


def train(t_net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, log_sigma=None, epo=200, patience=30, no_early_stopping=False):
    iter_loss = []
    vali_loss = []
    mae_history = []
    sae_history = []
    f1_history = []
    early_stopping_all = utils.EarlyStopping(logger, patience=patience, verbose=True)

    if config.dataAug:
        sigClass = utils.sigGen(config)

    path_all = os.path.join(modelDir, "All_best_onoff.ckpt")

    for e_i in range(epo):

        logger.info(f"# of epoches: {e_i}")
        for _, (_, _, X_scaled, Y_scaled, Y_of) in enumerate(tqdm(train_Dataloader)):
            if config.dataAug:
                X_scaled, Y_scaled, Y_of = utils.dataAug(X_scaled.clone(), Y_scaled.clone(), Y_of.clone(), sigClass, config)

            t_net.model_opt.zero_grad(set_to_none=True)

            X_scaled = X_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_scaled = Y_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_of = Y_of.type(torch.FloatTensor).to(device, non_blocking=True)

            y_pred_dish_r, y_pred_dish_c = t_net.model(X_scaled)

            loss_r = criterion[0](y_pred_dish_r,Y_scaled)
            loss_c = criterion[1](y_pred_dish_c, Y_of)

            if log_sigma is not None:
                # Uncertainty weighting: L = (1/2σ²)*L_task + log(σ)
                # Parameterized as log_sigma to keep σ > 0
                loss = (0.5 * torch.exp(-2 * log_sigma[0]) * loss_r + log_sigma[0] +
                        0.5 * torch.exp(-2 * log_sigma[1]) * loss_c + log_sigma[1])
            else:
                loss = loss_r + loss_c
            loss.backward()

            torch.nn.utils.clip_grad_norm_(t_net.model.parameters(), max_norm=1.0)
            t_net.model_opt.step()
            iter_loss.append(loss.item())

        epoch_losses = np.average(iter_loss)

        logger.info(f"Validation: ")
        maeScore, saeScore, f1Score, y_vali_ori, y_vali_pred_d_update, _, _, _ = utils.evaluateResult(net, config, vali_Dataloader, logger)
        val_loss = criterion[0](y_vali_ori, y_vali_pred_d_update)
        if log_sigma is not None:
            sigma_r = torch.exp(log_sigma[0]).item()
            sigma_c = torch.exp(log_sigma[1]).item()
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}, σ_reg={sigma_r:.4f}, σ_cls={sigma_c:.4f}.")
        else:
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}.")
        vali_loss.append(val_loss)
        mae_history.append(maeScore)
        sae_history.append(saeScore)
        f1_history.append(f1Score)

        if e_i % 10 == 0:
            checkpointName = os.path.join(modelDir, "checkpoint_" + str(e_i) + '.ckpt')
            utils.saveModel(logger, net, checkpointName)

        if not no_early_stopping:
            logger.info(f"Early stopping overall: ")
            early_stopping_all(np.mean(maeScore), net, path_all)
            if early_stopping_all.early_stop:
                print("Early stopping")
                break

    if no_early_stopping:
        utils.saveModel(logger, net, path_all)

    net_all = copy.deepcopy(net)
    checkpoint_all = torch.load(path_all, map_location=device)
    utils.loadModel(logger, net_all, checkpoint_all)
    net_all.model.eval()
    
    return net_all, mae_history, sae_history, f1_history

if __name__ == '__main__':
    args = get_args()
    utils.mkdir("log/" + args.subName)
    logger = utils.setup_log(args.subName, args.logname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computation device: {device}")
    logger.info(args)
    if args.debug:
        epo = 2
    else:
        epo = args.epochs

    # splitLoss = False
    # trainFull = True

    # Dataloder
    logger.info(f"loading data")
    train_data, val_data, test_data = utils.data_loader(args)

    logger.info(f"loading data finished")

    config_dict = {
        "input_size": 1,
        "batch_size": args.batch,
        "hidden": args.hidden,
        "lr": args.lr,
        "dropout": args.dropout,
        "logname": args.logname,
        "outputLength": args.outputLength,
        "inputLength" : args.inputLength,
        "subName": args.subName,
        "dataAug": args.dataAug,
        "prob0": args.prob0,
        "prob1": args.prob1,
        "prob2": args.prob2,
        "prob3": args.prob3,
    }

    config = TrainConfig.from_dict(config_dict)
    modelDir = utils.mkdirectory(config.subName, saveModel=True)
    joblib.dump(config, os.path.join(modelDir, "config.pkl"))


    logger.info(f"Training size: {train_data.cumulative_sizes[-1]:d}.")

    index = np.arange(0,train_data.cumulative_sizes[-1])
    train_subsampler = torch.utils.data.SubsetRandomSampler(index)
    train_Dataloader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=train_subsampler,
        num_workers=1,
        pin_memory=True)

    sampler = utils.testSampler(val_data.cumulative_sizes[-1], config.outputLength)
    sampler_test = utils.testSampler(test_data.cumulative_sizes[-1], config.outputLength)

    vali_Dataloader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=1,
        pin_memory=True)

    test_Dataloader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        sampler=sampler_test,
        num_workers=1,
        pin_memory=True)

    logger.info("Initialize model")
    model = MAT(config).to(device)
    logger.info("Model MAT")

    # Learnable log-sigma parameters for uncertainty weighting (Kendall et al. 2018)
    # log_sigma[0] -> regression task, log_sigma[1] -> classification task
    log_sigma = nn.Parameter(torch.zeros(2, device=device))

    optim = optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad] + [log_sigma],
        lr=config.lr
    )
    net = Basic(model, optim)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(modelDir, args.checkpoint)
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net = utils.loadModel(logger, net, checkpoint)
        else:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            logger.info("Starting training from scratch")
    
    criterion_r = nn.MSELoss()
    criterion_c = nn.BCELoss()
    criterion = [criterion_r, criterion_c]

    logger.info("Training start")
    net_all, mae_history, sae_history, f1_history = train(net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, log_sigma=log_sigma, epo=epo, patience=args.patience, no_early_stopping=args.no_early_stopping)
    logger.info("Training end")

    plot_metrics(mae_history, sae_history, f1_history, modelDir)

    logger.info("validation start")
    utils.evaluateResult(net_all, config, vali_Dataloader, logger)
    logger.info("test start")
    utils.evaluateResult(net_all, config, test_Dataloader, logger)
