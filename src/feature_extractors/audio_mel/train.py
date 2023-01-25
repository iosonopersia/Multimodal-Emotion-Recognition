import os
import torch
import wandb
from dataset import Dataset
from model import AudioMelFeatureExtractor
from losses.M2FNetAudioEmbeddingLoss import M2FNetAudioEmbeddingLoss
from tqdm import tqdm
from datetime import datetime
from munch import Munch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import sklearn

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main(config=None):
    #CONFIG
    with open('./src/feature_extractors/audio_mel/config_audio_mel.yaml', 'rt', encoding='utf-8') as f:
        config = Munch.fromYAML(f.read())

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    # TRAIN DATA
    data_train = Dataset(mode="train", config=config)

    # VAL DATA
    data_val = Dataset(mode="val", config=config)

    #============MODEL===============
    #--------------------------------
    model = AudioMelFeatureExtractor().to(device)

    #============CRITERION===============
    #------------------------------------

    #triplet loss
    adaptive = config.solver.adaptive_triplet_margin_loss
    covariance = config.solver.covariance_loss
    variance = config.solver.variance_loss
    criterion = M2FNetAudioEmbeddingLoss(
        adaptive=adaptive,
        covariance_enabled=covariance,
        variance_enabled=variance).to(device)

    #============OPTIMIZER===============
    #------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.solver.lr, weight_decay=config.solver.weight_decay)

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

    if (load_checkpoint):
        if os.path.exists(load_checkpoint_path):
            checkpoint = torch.load(load_checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1 # start from the next epoch
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise ValueError("Checkpoint not found")

    #============WRITER==============
    if config.wandb.enabled:
        entity_name = config.wandb.entity
        resume_run = config.wandb.resume_run
        resume_run_id = config.wandb.resume_run_id
        os_start_method = 'spawn' if os.name == 'nt' else 'fork'
        run_datetime = datetime.now().isoformat().split('.')[0]
        wandb.init(
            project=config.wandb.project_name,
            name=run_datetime,
            config=config,
            settings=wandb.Settings(start_method=os_start_method),
            entity = entity_name,
            resume="must" if resume_run else False,
            id=resume_run_id)

    #============TRAIN===============
    #--------------------------------
    if config.DEBUG.train:
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

        if config.checkpoint.load_checkpoint:
            checkpoint_folder = os.path.dirname(config.checkpoint.load_path)
            best_weights_path = os.path.join(os.path.abspath(checkpoint_folder), 'best_weights.pth')
            if os.path.exists(best_weights_path):
                best_weights = torch.load(best_weights_path)
                min_loss_val = best_weights['min_loss_val']
                patience_counter = start_epoch - (best_weights['epoch'] + 1)
                patience_counter = max(0, patience_counter)

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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_checkpoint_path)

        # visualize_model(model, dl_test, device, config.DEBUG.visualization_type, epoch=epoch, save=True, visualize=False, wandb_log=wandb_log)

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
                        'min_loss_val': min_loss_val,
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
                            'epoch': best_model["epoch"],
                            'model_state_dict': best_model["model_state_dict"],
                            'optimizer_state_dict': best_model["optimizer_state_dict"],
                            }, save_checkpoint_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break

    if wandb_log:
        wandb.finish()

    return {'loss_values': losses_values}

def train(model, data_train, criterion, optimizer, epoch, wandb_log, device):
    loss_train = 0.0
    batch_size = data_train.config.train.data_loader.batch_size
    n_steps = len(data_train) // batch_size
    # n_steps = 300
    model.eval()
    for idx_batch in tqdm(range(n_steps), desc=f"Epoch {epoch}"):
        with torch.inference_mode():
            data = data_train.get_batched_triplets(batch_size, model, mining_type="hard", device=device)

        anchor = data["anchor"].to(device)
        positive = data["positive"].to(device)
        negative = data["negative"].to(device)

        # Feature extractor
        optimizer.zero_grad()
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

        loss.backward()
        optimizer.step()

        loss_train += loss.item()

        if wandb_log:
            running_loss = loss_train / (idx_batch + 1)
            global_step = epoch * n_steps + idx_batch
            wandb.log({'Train_loss': running_loss, 'Global_step': global_step})

    return loss_train / n_steps

def validate(model, data_val, criterion, device):
    loss_eval = 0.0
    batch_size = data_val.config.val.data_loader.batch_size
    n_steps = len(data_val) // batch_size

    model.eval()
    with torch.inference_mode():
        for _ in tqdm(range(n_steps), "Validation"):
            data = data_val.get_batched_triplets(batch_size, model, mining_type="hard", device=device)
            anchor = data["anchor"].to(device)
            positive = data["positive"].to(device)
            negative = data["negative"].to(device)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

            loss_eval += loss.item()

    return loss_eval / n_steps

def visualize_model(model, dataloader, device, visualization_type="3D", epoch=0, save=True, visualize=False, wandb_log=False):
    if visualization_type == "3D":
        tsne = TSNE(n_components=3)
    elif visualization_type == "2D":
        tsne = TSNE(n_components=2)
    else:
        raise ValueError("Visualization type not supported")

    model.eval()
    embeddings = []
    true_labels = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Visualizing", total=len(dataloader)):
            inputs = batch["audio_mel_spectogram"].to(device)
            outputs = model(inputs)
            embeddings.extend(outputs.cpu().tolist())
            true_labels.extend(batch["emotion"].squeeze(dim=-1).tolist())
    embeddings = np.array(embeddings, dtype=np.float32)
    true_labels = np.array(true_labels, dtype=np.int32)

    #silhouette score
    silhouette_score = sklearn.metrics.silhouette_score(embeddings, true_labels)
    print(f"Silhouette score: {silhouette_score}")

    # visualize embeddings
    embeddings = PCA(random_state=0).fit_transform(embeddings)[:,:50]
    embeddings = tsne.fit_transform(embeddings)

    x = embeddings[:, 0]
    y = embeddings[:, 1]
    if visualization_type == "3D":
        z = embeddings[:, 2]

    # Create a scatter plot of the 3D embeddings
    if visualization_type == "3D":
        fig = px.scatter_3d(x=x, y=y, z=z, color=true_labels, opacity=0.7, width=800, height=800)
    else:
        fig = px.scatter(x=x, y=y, color=true_labels, opacity=0.7, width=800, height=800)

    # Disable hover data
    fig.update_traces(hovertemplate=None, hoverinfo="skip")

    save_dir_png = os.path.join("src","feature_extractors","audio_mel", "visualization", "png")
    save_dir_html = os.path.join("src","feature_extractors","audio_mel", "visualization", "html")

    # check if directory exists and create it if doesn't
    if not os.path.exists(save_dir_png):
        os.makedirs(save_dir_png)
    if not os.path.exists(save_dir_html):
        os.makedirs(save_dir_html)
    if save:
        fig.write_html(os.path.join(save_dir_html, f"visualization_{epoch}.html"))
        fig.write_image(os.path.join(save_dir_png, f"visualization_{epoch}.png"))
        if wandb_log:
            #png
            wandb.log({"Visualization_png": [wandb.Image(os.path.join(save_dir_png, f"visualization_{epoch}.png"))]})
            #html
            wandb.log({"Visualization_html": [wandb.Html(os.path.join(save_dir_html, f"visualization_{epoch}.html"))]})

    if visualize:
        fig.show()

if __name__ == "__main__":
    main()
