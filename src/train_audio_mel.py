import os
import torch
import wandb
from datasets.datasetAudioMel import DatasetMelAudio as Dataset
from models.AudioMelFeatureExtractor import AudioMelFeatureExtractor
from tqdm import tqdm
from datetime import datetime
from sklearn.utils import class_weight
from munch import Munch
from losses.AdaptiveTripletMarginLoss import AdaptiveTripletMarginLoss

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main(config=None):
    #CONFIG
    with open('./src/config_audio_mel.yaml', 'rt', encoding='utf-8') as f:
        config = Munch.fromYAML(f.read())

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    # TRAIN DATA
    data_train = Dataset(mode="train", config=config)

    # train_dl_cfg = config.train.data_loader
    # dl_train = torch.utils.data.DataLoader(data_train, **train_dl_cfg)

    # VAL DATA
    data_val = Dataset(mode="val", config=config)

    # val_dl_cfg = config.val.data_loader
    # dl_val = torch.utils.data.DataLoader(data_val, **val_dl_cfg)

    #============MODEL===============
    #--------------------------------
    model = AudioMelFeatureExtractor().to(device)

    #============CRITERION===============
    #------------------------------------

    #triplet loss
    if config.solver.adaptive_triplet_margin_loss.enabled:
        criterion = AdaptiveTripletMarginLoss()
    else:
        criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    #============OPTIMIZER===============
    #------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.solver.lr, weight_decay=config.solver.weight_decay)

    #============WRITER==============
    if config.wandb.enabled:
        os_start_method = 'spawn' if os.name == 'nt' else 'fork'
        run_datetime = datetime.now().isoformat().split('.')[0]
        wandb.init(
            project=config.wandb.project_name,
            name=run_datetime,
            config=config,
            settings=wandb.Settings(start_method=os_start_method))

    #============SCHEDULER===============
    #------------------------------------
    lr_scheduler = None
    if config.solver.scheduler.enabled:
        if (config.solver.scheduler.scheduler_fn == 'ExponentialLR'):
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.solver.scheduler.gamma)
        else:
            raise ValueError("Scheduler not supported")

    #============LOAD================
    #--------------------------------
    start_epoch = 0
    load_checkpoint = config.checkpoint.load_checkpoint
    load_checkpoint_path = config.checkpoint.load_path

    if (load_checkpoint and os.path.exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #============TRAIN===============
    #--------------------------------
    print("Training...")
    training_loop(
        model,
        data_train,
        data_val,
        criterion,
        optimizer,
        lr_scheduler,
        start_epoch,
        config,
        device,
        # hyperparameter_search
    )
    print("Training complete")


def training_loop(model, data_train, data_val, criterion, optimizer, lr_scheduler, start_epoch, config, device):
    losses_values = []
    val_losses_values = []

    solver_cfg = config.solver
    early_stopping = solver_cfg.early_stopping.enabled
    epochs = solver_cfg.epochs
    use_scheduler = solver_cfg.scheduler.enabled
    restore_best_weights = solver_cfg.early_stopping.restore_best_weights
    patience = solver_cfg.early_stopping.patience

    wandb_cfg = config.wandb
    wandb_log = wandb_cfg.enabled

    checkpoint_cfg = config.checkpoint
    save_checkpoint = checkpoint_cfg.save_checkpoint
    save_checkpoint_path = checkpoint_cfg.save_path

    if wandb_log and wandb_cfg.watch_model:
        wandb.watch(
            model,
            criterion=criterion,
            log="all", # default("gradients"), "parameters", "all"
            log_freq=100,
            log_graph=False)

    if early_stopping:
        min_loss_val = float('inf')
        patience_counter = 0

    for epoch in range(start_epoch, epochs):
        loss_train = train(
            model,
            data_train,
            criterion,
            optimizer,
            epoch,
            wandb_log,
            device)
        losses_values.append(loss_train)

        loss_val = validate(
            model,
            data_val,
            criterion,
            device)
        val_losses_values.append(loss_val)

        if save_checkpoint:
            os.makedirs(save_checkpoint_path.rsplit("/", 1)[0], exist_ok=True)
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_checkpoint_path)

        lr = optimizer.param_groups[0]['lr']
        if use_scheduler:
            lr_scheduler.step()

        print(f'Epoch: {epoch} '
              f' Lr: {lr:.8f} '
              f' Loss: Train = [{loss_train:.3E}] - Val = [{loss_val:.3E}]')

        if wandb_log:
            wandb.log({'Learning_Rate': lr, 'Train': loss_train, 'Validation': loss_val, 'Epoch': epoch})

        # Early stopping
        if early_stopping:
            if loss_val < min_loss_val:
                min_loss_val = loss_val
                patience_counter = 0
                if restore_best_weights:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_checkpoint_path.rsplit("/", 1)[0]+"/best_weights.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: patience {patience} reached")
                    if restore_best_weights:
                        best_model = torch.load(save_checkpoint_path.rsplit("/", 1)[0]+"/best_weights.pth")
                        torch.save({
                            'epoch': best_model["epoch"]+1,
                            'model_state_dict': best_model["model_state_dict"],
                            'optimizer_state_dict': best_model["optimizer_state_dict"],
                            }, save_checkpoint_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break

    if wandb_log:
        wandb.finish()

    return {'loss_values': losses_values}

def train(model, data_train, criterion, optimizer, epoch, wandb_log, device):
    loss_train = 0
    batch_size = data_train.config.train.data_loader.batch_size
    n_steps = len(data_train) // batch_size
    model.train()
    for idx_batch in tqdm(range(n_steps), "Training epoch {}".format(epoch)):
        with torch.inference_mode():
            data = data_train.get_batched_triplets(batch_size, model)

        anchor, positive, negative = data["anchor"].to(device), data["positive"].to(device), data["negative"].to(device)

        # Feature extractor
        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_train += loss.item()

        if wandb_log:
            running_loss = loss_train / (idx_batch + 1)
            global_step = epoch * n_steps + idx_batch
            wandb.log({'Train_loss': running_loss, 'Global_step': global_step})

    return loss_train / n_steps

def validate(model, data_val, criterion, device):
    loss_eval = 0
    batch_size = data_val.config.val.data_loader.batch_size
    n_steps = len(data_val) // batch_size
    model.eval()

    with torch.inference_mode():
        for _ in tqdm(range(n_steps), "Validation"):
            data = data_val.get_batched_triplets(batch_size, model)
            anchor, positive, negative = data["anchor"].to(device), data["positive"].to(device), data["negative"].to(device)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

            loss_eval += loss.item()

    return loss_eval / n_steps


if __name__ == "__main__":
    main()
