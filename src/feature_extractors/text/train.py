import os
import torch
import wandb
from dataset import Dataset, collate_fn
from model import TextERC
from tqdm import tqdm
from datetime import datetime
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score
from transformers.optimization import get_constant_schedule_with_warmup
from utils import get_config

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
    dl_train = torch.utils.data.DataLoader(data_train, collate_fn=collate_fn, **train_dl_cfg)

    # VAL DATA
    data_val = Dataset(mode="val")
    val_dl_cfg = config.val.data_loader
    dl_val = torch.utils.data.DataLoader(data_val, collate_fn=collate_fn, **val_dl_cfg)

    # TEST DATA
    data_test = Dataset(mode="test")
    test_dl_cfg = config.test.data_loader
    dl_test = torch.utils.data.DataLoader(data_test, collate_fn=collate_fn, **test_dl_cfg)

    #============MODEL===============
    #--------------------------------
    model = TextERC().to(device)

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
    finetuning_lr = config.solver.finetuning_lr
    frozen_lr = config.solver.frozen_lr
    weight_decay = config.solver.weight_decay
    frozen_epochs_optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=frozen_lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetuning_lr, weight_decay=weight_decay)

    #============WANDB==============
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
    warmup_epochs = config.solver.warmup_epochs
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_epochs*len(dl_train))

    #============TRAIN===============
    #--------------------------------
    print("Training...")
    training_loop(
        model,
        dl_train,
        dl_val,
        dl_test,
        criterion,
        optimizer,
        frozen_epochs_optimizer,
        lr_scheduler,
        config,
        device
    )
    print("Training complete")


def training_loop(model, dl_train, dl_val, dl_test, criterion, optimizer, frozen_epochs_optimizer, lr_scheduler, config, device):
    losses_values = []
    val_losses_values = []

    solver_cfg = config.solver
    epochs = solver_cfg.epochs
    num_frozen_epochs = solver_cfg.num_frozen_epochs

    wandb_cfg = config.wandb
    wandb_log = wandb_cfg.enabled

    early_stopping = solver_cfg.early_stopping.enabled
    restore_best_weights = solver_cfg.early_stopping.restore_best_weights
    patience = solver_cfg.early_stopping.patience

    #============CHECKPOINT===============
    checkpoint_cfg = config.checkpoint
    save_checkpoint = checkpoint_cfg.save_checkpoint
    save_checkpoint_path = os.path.abspath(checkpoint_cfg.save_path)
    os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True) # Create folder if it doesn't exist

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

    for epoch in range(epochs):
        is_frozen_epoch = epoch < num_frozen_epochs
        if is_frozen_epoch:
            model.freeze()
            current_optimizer = frozen_epochs_optimizer
        else:
            model.unfreeze()
            current_optimizer = optimizer

        loss_train = train(
            model,
            dl_train,
            criterion,
            current_optimizer,
            lr_scheduler,
            is_frozen_epoch,
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

        loss_test, accuracy_test, weighted_f1_test = validate(
            model,
            dl_test,
            criterion,
            device)

        if save_checkpoint:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, save_checkpoint_path)

        print(f'Epoch: {epoch} Train=[{loss_train:.3E}] Val=[{loss_val:.3E}] Accuracy=[{accuracy * 100:.3f}%] Weighted_F1=[{weighted_f1 * 100:.3f}%]')

        if wandb_log:
            wandb.log({
                'Params/Epoch': epoch,
                'Train/Loss': loss_train,
                'Validation/Loss': loss_val,
                'Validation/Accuracy': accuracy,
                'Validation/Weighted_F1': weighted_f1,
                'Test/Loss': loss_test,
                'Test/Accuracy': accuracy_test,
                'Test/Weighted_F1': weighted_f1_test,
            })

        # Early stopping
        if early_stopping:
            if loss_val < min_loss_val:
                min_loss_val = loss_val
                patience_counter = 0
                if restore_best_weights:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
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

    if wandb_log:
        wandb.finish()

    return {'loss_values': losses_values}

def train(model, dl_train, criterion, optimizer, lr_scheduler, is_frozen_epoch, epoch, wandb_log, device):
    loss_train = 0

    model.train()
    for idx_batch, batch in tqdm(enumerate(dl_train), total=len(dl_train)):
        text = batch["text"].to(device)
        emotion = batch["emotion"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(text, attention_mask)
        loss = criterion(outputs, emotion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if not is_frozen_epoch:
            lr_scheduler.step()
        loss_train += loss.item()

        if wandb_log:
            lr = optimizer.param_groups[0]['lr']
            running_loss = loss_train / (idx_batch + 1)
            global_step = epoch * len(dl_train) + idx_batch
            wandb.log({
                'Params/Learning_Rate': lr,
                'Train/Running_loss': running_loss,
                'Params/Global_step': global_step})

    return loss_train / len(dl_train)

def validate(model, dl_val, criterion, device):
    loss_eval = 0
    accuracy = 0
    weighted_f1 = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dl_val, total=len(dl_val)):
            text = batch["text"].to(device)
            emotion = batch["emotion"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(text, attention_mask)
            loss = criterion(outputs, emotion)

            # Calculate metrics
            emotion_predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            emotion = emotion.cpu().numpy()
            accuracy += accuracy_score(emotion, emotion_predicted)
            weighted_f1 += f1_score(emotion, emotion_predicted, average='weighted')

            loss_eval += loss.item()

    num_batches = len(dl_val)
    return loss_eval/num_batches, accuracy/num_batches, weighted_f1/num_batches


if __name__ == "__main__":
    main()
