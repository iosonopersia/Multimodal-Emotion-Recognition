from genericpath import exists
import os
import torch
import wandb
from datasets.datasetAudioMel import DatasetMelAudio as Dataset
from models.AudioMelFeatureExtractor import TestAudioExtractor
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from munch import Munch

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

    train_dl_cfg = config.train.data_loader
    dl_train = torch.utils.data.DataLoader(data_train, **train_dl_cfg)

    # VAL DATA
    data_val = Dataset(mode="val", config=config)

    val_dl_cfg = config.val.data_loader
    dl_val = torch.utils.data.DataLoader(data_val, **val_dl_cfg)

    #============MODEL===============
    #--------------------------------
    model = TestAudioExtractor().to(device)

    #============CRITERION===============
    #------------------------------------
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

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

    if (load_checkpoint and exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #============TRAIN===============
    #--------------------------------
    print("Training...")
    training_loop(
        model,
        dl_train,
        dl_val,
        criterion,
        optimizer,
        lr_scheduler,
        start_epoch,
        config,
        device,
        # hyperparameter_search
    )
    print("Training complete")


def training_loop(model, dl_train, dl_val, criterion, optimizer, lr_scheduler, start_epoch, config, device):
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

    #============CHECKPOINT===============
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


    for epoch in range(start_epoch, epochs):
        loss_train = train(
            model,
            dl_train,
            criterion,
            optimizer,
            epoch,
            wandb_log,
            device)
        losses_values.append(loss_train)

        loss_val, accuracy, weighted_f1 = validate(
            model,
            dl_val,
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
              f' Loss: Train = [{loss_train:.3E}] - Val = [{loss_val:.3E}] - accuracy = [{accuracy * 100:.3f}%] - weighted_f1 = [{weighted_f1 * 100:.3f}%]')

        if wandb_log:
            wandb.log({'Learning_Rate': lr, 'Train': loss_train, 'Validation': loss_val, 'Epoch': epoch, 'accuracy': accuracy, 'weighted_f1': weighted_f1})


        # Hyperparameter search
        # if hyperparameter_search:
        #     with tune.checkpoint_dir(epoch):
        #         path = save_checkpoint_path
        #         os.makedirs(path, exist_ok=True)
        #         path += os.sep + "checkpoint.pth"
        #         torch.save((model.state_dict(), optimizer.state_dict()), path)

        #     tune.report(loss=orig_mre_loss)

    if wandb_log:
        wandb.finish()

    return {'loss_values': losses_values}

def train(model, dl_train, criterion, optimizer, epoch, wandb_log, device):
    loss_train = 0

    model.train()
    for idx_batch, data in tqdm(enumerate(dl_train), total=len(dl_train)):
        audio, emotion = data["audio_mel_spectogram"].to(device), data["emotion"].to(device)


        # Feature extractor
        optimizer.zero_grad()
        # outputs = model(text, audio, mask)
        outputs = model(audio)
        loss = criterion(outputs, emotion.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_train += loss.item()

        if wandb_log:
            running_loss = loss_train / (idx_batch + 1)
            global_step = epoch * len(dl_train) + idx_batch
            wandb.log({'Train_loss': running_loss, 'Global_step': global_step})

    return loss_train / len(dl_train)

def validate(model, dl_val, criterion, device):
    loss_eval = 0
    accuracy = 0
    weighted_f1 = 0

    model.eval()
    with torch.inference_mode():
        for data in tqdm(dl_val, total=len(dl_val)):
            audio, emotion = data["audio_mel_spectogram"].to(device), data["emotion"].to(device)

            # outputs = model(text, audio, mask)
            outputs = model(audio)
            loss = criterion(outputs, emotion.squeeze())

            # Calculate metrics
            emotion_predicted = torch.argmax(outputs, dim=1)
            # mask = (emotion != -1)
            # emotion_predicted = emotion_predicted[mask].flatten().cpu().numpy()
            # emotion = emotion[mask].flatten().cpu().numpy()
            accuracy += accuracy_score(emotion.squeeze().cpu().numpy(), emotion_predicted.cpu().numpy())
            weighted_f1 += f1_score(emotion.squeeze().cpu().numpy(), emotion_predicted.cpu().numpy(), average='weighted')

            loss_eval += loss.item()

    return loss_eval/len(dl_val), accuracy/len(dl_val), weighted_f1/len(dl_val)


if __name__ == "__main__":
    main()
