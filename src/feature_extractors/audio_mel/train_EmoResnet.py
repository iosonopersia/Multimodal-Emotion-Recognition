import os
import torch
import wandb
from dataset import Dataset as Dataset
from AudioMelFeatureExtractor import EmoResnet
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from munch import Munch
from sklearn.utils import class_weight

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main(config=None):
    #CONFIG
    with open('./src/feature_extractors/audio_mel/config_emoResNet.yaml', 'rt', encoding='utf-8') as f:
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
    model = EmoResnet().to(device)

    #============CRITERION===============
    #------------------------------------
    balance_classes = False
    if balance_classes:
        emotions = data_train.get_labels() # Use training data to compute class weights
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1, 2, 3, 4, 5, 6], y=emotions)
        class_weights = torch.as_tensor(class_weights, dtype=torch.float, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    else:
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

    if early_stopping:
        best_weights_save_path = os.path.join(os.path.dirname(save_checkpoint_path), f'best_weights.pth')
        min_loss_val = float('inf')
        patience_counter = 0


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
            'model_state_dict': model.resnet18.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_checkpoint_path)

                    # Early stopping
        if early_stopping:
            if loss_val < min_loss_val:
                min_loss_val = loss_val
                patience_counter = 0
                if restore_best_weights:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.resnet18.state_dict(),
                    }, best_weights_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: patience {patience} reached")
                    if restore_best_weights:
                        best_model = torch.load(best_weights_save_path)
                        torch.save({
                            'epoch': best_model['epoch'] + 1,
                            'model_state_dict': best_model['model_state_dict'],
                        }, save_checkpoint_path)
                        os.remove(best_weights_save_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break

        lr = optimizer.param_groups[0]['lr']
        if use_scheduler:
            lr_scheduler.step()

        print(f'Epoch: {epoch} '
              f' Lr: {lr:.8f} '
              f' Loss: Train = [{loss_train:.3E}] - Val = [{loss_val:.3E}] - accuracy = [{accuracy * 100:.3f}%] - weighted_f1 = [{weighted_f1 * 100:.3f}%]')

        if wandb_log:
            wandb.log({'Learning_Rate': lr, 'Train': loss_train, 'Validation': loss_val, 'Epoch': epoch, 'accuracy': accuracy, 'weighted_f1': weighted_f1})

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
            accuracy += accuracy_score(emotion.squeeze().cpu().numpy(), emotion_predicted.cpu().numpy())
            weighted_f1 += f1_score(emotion.squeeze().cpu().numpy(), emotion_predicted.cpu().numpy(), average='weighted')

            loss_eval += loss.item()

    return loss_eval/len(dl_val), accuracy/len(dl_val), weighted_f1/len(dl_val)


if __name__ == "__main__":
    main()
