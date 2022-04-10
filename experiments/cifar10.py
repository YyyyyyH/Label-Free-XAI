import os

import pandas as pd
import torch
import hydra
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models.images import SimCLR
from omegaconf import DictConfig
from pathlib import Path
from torchvision.models import resnet18, resnet34
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, GaussianBlur
from torch.utils.data import DataLoader
from explanations.features import attribute_auxiliary
from utils.feature_attribution import generate_masks
from captum.attr import GradientShap, IntegratedGradients
from tqdm import tqdm


@hydra.main(config_name='simclr_config.yaml', config_path=str(Path.cwd()/'models'))
def consistency_feature_importance(args: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pert_percentages = [5, 10, 20, 50, 70, 80, 90, 100]
    perturbation = GaussianBlur(3, sigma=1).to(device)

    # Prepare model
    torch.manual_seed(args.seed)
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).to(device)
    logging.info(f'Base model: {args.backbone} - feature dim: {model.feature_dim} - projection dim {args.projection_dim}')
    logging.info('Fitting SimCLR model')
    #model.fit(args, device)
    model.load_state_dict(torch.load("simclr_resnet18_epoch100.pt"), strict=False)
    # Compute feature importance
    W = 32
    test_batch_size = int(args.batch_size/20)
    encoder = model.enc
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    test_set = CIFAR10(data_dir, False, transform=ToTensor())
    test_loader = DataLoader(test_set, test_batch_size)
    attr_methods = {"Gradient Shap": GradientShap,
                    "Integrated Gradients": IntegratedGradients,
                    "Random": None}
    results_data = []
    for method_name in attr_methods:
        logging.info(f'Computing feature importance with {method_name}')
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(encoder, test_loader, device, GradientShap(encoder), perturbation)
        else:
            np.random.seed(args.seed)
            attr = np.random.randn(len(test_set), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(f'Perturbing {pert_percentage}% of the features with {method_name}')
            mask_size = int(pert_percentage*W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[batch_id*test_batch_size:batch_id*test_batch_size+len(images)].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask*images + (1-mask)*perturbation(images)
                pert_reps = encoder(images)
                rep_shift = torch.mean(torch.sum((original_reps-pert_reps)**2, dim=-1)).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    logging.info(f"Saving the plot")
    results_df = pd.DataFrame(results_data, columns=["Method", "% of features perturbed", "Representation Shift"])
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(data=results_df, x="% of features perturbed", y="Representation Shift", hue="Method")
    plt.tight_layout()
    plt.savefig("cifar10_consistency_features.pdf")
    plt.close()


if __name__ == '__main__':
    consistency_feature_importance()