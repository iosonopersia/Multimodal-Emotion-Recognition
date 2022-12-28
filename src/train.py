from genericpath import exists
import os
import torch
import wandb
from utils import get_config
from datasets.dataset import Dataset
from models.FeatureExtractor import FeatureExtractor
from models.M2FNet import M2FNet
from tqdm import tqdm
from datetime import datetime
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main(config=None):
    #CONFIG
    config = get_config()

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    # TRAIN DATA
    data_train = Dataset(mode="train")

    train_dl_cfg = config.train.data_loader
    dl_train = torch.utils.data.DataLoader(data_train, collate_fn=data_train.my_collate_fn, **train_dl_cfg)

    # VAL DATA
    data_val = Dataset(mode="val")

    val_dl_cfg = config.val.data_loader
    dl_val = torch.utils.data.DataLoader(data_val, collate_fn=data_val.my_collate_fn, **val_dl_cfg)

    #============MODEL===============
    #--------------------------------
    feature_embedding_model = FeatureExtractor().to(device)
    model = M2FNet(config.model).to(device)

    #============CRITERION===============
    #------------------------------------
    criterion = config.solver.loss_fn
    balance_classes = config.solver.balance_classes
    if criterion == "CE":
        if balance_classes:
            emotions = data_train.get_labels() # Use training data to compute class weights
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1, 2, 3, 4, 5, 6], y=emotions)
            class_weights = torch.as_tensor(class_weights, dtype=torch.float, device=device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        raise ValueError("Criterion not supported")

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
        feature_embedding_model,
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


def training_loop(model, feature_embedding_model, dl_train, dl_val, criterion, optimizer, lr_scheduler, start_epoch, config, device):
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
    save_checkpoint_folder = checkpoint_cfg.save_folder
    save_checkpoint_path = os.path.join(os.path.abspath(save_checkpoint_folder), f'checkpoint.pth')
    os.makedirs(save_checkpoint_folder, exist_ok=True) # Create folder if it doesn't exist
    for file in os.listdir(save_checkpoint_folder): # Delete all files in folder
                os.remove(os.path.join(save_checkpoint_folder, file))

    if wandb_log and wandb_cfg.watch_model:
        wandb.watch(
            model,
            criterion=criterion,
            log="all", # default("gradients"), "parameters", "all"
            log_freq=100,
            log_graph=False)

    if early_stopping:
        best_weights_save_path = os.path.join(os.path.abspath(save_checkpoint_folder), f'best_weights.pth')
        min_loss_val = float('inf')
        patience_counter = 0

    for epoch in range(start_epoch, epochs):
        loss_train = train(
            model,
            feature_embedding_model,
            dl_train,
            criterion,
            optimizer,
            epoch,
            wandb_log,
            device)
        losses_values.append(loss_train)

        loss_val, accuracy, weighted_f1 = validate(
            model,
            feature_embedding_model,
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
              f' Loss: Train = [{loss_train:.3E}] - Val = [{loss_val:.3E}] - accuracy = [{accuracy * 100:.3f}%] - weighted_f1 = [{weighted_f1 * 100:.3f}]')

        if wandb_log:
            wandb.log({'Learning_Rate': lr, 'Train': loss_train, 'Validation': loss_val, 'Epoch': epoch, 'accuracy': accuracy, 'weighted_f1': weighted_f1})

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
                        }, best_weights_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: patience {patience} reached")
                    if restore_best_weights:
                        best_model = torch.load(best_weights_save_path)
                        torch.save({
                            'epoch': best_model["epoch"] + 1,
                            'model_state_dict': best_model["model_state_dict"],
                            'optimizer_state_dict': best_model["optimizer_state_dict"],
                            }, save_checkpoint_path)
                        os.remove(best_weights_save_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break

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

def train(model, feature_embedding_model, dl_train, criterion, optimizer, epoch, wandb_log, device):
    loss_train = 0

    model.train()
    for idx_batch, data in tqdm(enumerate(dl_train), total=len(dl_train)):
        text, audio, emotion = data["text"], data["audio"], data["emotion"]
        emotion = emotion.to(device)

        feature_embedding_model.eval()
        with torch.no_grad():
            text = [t.to(device) for t in text]
            audio = [[aa.to(device) for aa in a] for a in audio]

            text, audio, mask = feature_embedding_model(text, audio)

        # Start recording gradients from here
        text.requires_grad_(True)
        audio.requires_grad_(True)

        # Feature extractor
        optimizer.zero_grad()
        outputs = model(text, audio, mask)
        loss = criterion(outputs.permute(0, 2, 1), emotion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_train += loss.item()

        if wandb_log:
            running_loss = loss_train / (idx_batch + 1)
            global_step = epoch * len(dl_train) + idx_batch
            wandb.log({'Train_loss': running_loss, 'Global_step': global_step})

    return loss_train / len(dl_train)

def validate(model, feature_embedding_model, dl_val, criterion, device):
    loss_eval = 0
    accuracy = 0
    weighted_f1 = 0

    feature_embedding_model.eval()
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dl_val, total=len(dl_val)):
            text, audio, emotion = data["text"], data["audio"], data["emotion"]
            emotion = emotion.to(device)

            text = [t.to(device) for t in text]
            audio = [[aa.to(device) for aa in a] for a in audio]
            text, audio, mask = feature_embedding_model(text, audio)

            outputs = model(text, audio, mask)
            loss = criterion(outputs.permute(0, 2, 1), emotion)

            # Calculate metrics
            emotion_predicted = torch.argmax(outputs, dim=2)
            mask = (emotion != -1)
            emotion_predicted = emotion_predicted[mask].flatten().cpu().numpy()
            emotion = emotion[mask].flatten().cpu().numpy()
            accuracy += accuracy_score(emotion, emotion_predicted)
            weighted_f1 += f1_score(emotion, emotion_predicted, average='weighted')

            loss_eval += loss.item()

    return loss_eval/len(dl_val), accuracy/len(dl_val), weighted_f1/len(dl_val)


if __name__ == "__main__":
    main()
