import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class MultiTaskModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, infection_class_weights, organ_class_weights, lr=0.001):
        super(MultiTaskModel, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Task-specific layers
        self.infection_head = nn.Linear(hidden_dim, 1)
        self.organ_dysfunction_head = nn.Linear(hidden_dim, 1)
        # Store class weights as tensors
        self.infection_class_weights = torch.tensor(infection_class_weights, dtype=torch.float32)
        self.organ_class_weights = torch.tensor(organ_class_weights, dtype=torch.float32)
        # Loss function
        self.criterion = nn.BCELoss(reduction="none")  # No reduction to apply weights manually

    def forward(self, x):
        shared_output = self.shared_layer(x)
        infection_output = torch.sigmoid(self.infection_head(shared_output))
        organ_dysfunction_output = torch.sigmoid(self.organ_dysfunction_head(shared_output))
        return infection_output, organ_dysfunction_output

    def compute_loss(self, output, target, class_weights):
        # Compute sample weights
        sample_weights = class_weights[0] * (1 - target) + class_weights[1] * target
        # Compute BCE loss without reduction
        loss = self.criterion(output.squeeze(), target.float())
        # Apply sample weights
        weighted_loss = sample_weights * loss
        return weighted_loss.mean()

    def calculate_safe_metrics(self, y_true, y_pred):
        """
        Safely calculate precision, recall, F1-score.
        Avoids warnings when one class is missing in the batch.
        """
        if len(torch.unique(y_true)) < 2:  # Only one class present in the batch
            return 0.0, 0.0, 0.0  # Precision, Recall, F1 = 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.cpu(), y_pred.cpu(), average='binary', zero_division=0
        )
        return precision, recall, f1

    def calculate_auroc_safe(self, y_true, y_pred):
        """
        Safely calculate AUROC. If only one class is present, return None.
        """
        if len(torch.unique(y_true)) < 2:  # Only one class in y_true
            return None
        return roc_auc_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    def training_step(self, batch, batch_idx):
        x, y_infection, y_organ, y_sepsis = batch
        infection_output, organ_output = self(x)
        sepsis_prob = infection_output.squeeze() * organ_output.squeeze()

        # Compute weighted losses
        infection_loss = self.compute_loss(infection_output, y_infection, self.infection_class_weights.to(self.device))
        organ_loss = self.compute_loss(organ_output, y_organ, self.organ_class_weights.to(self.device))
        loss = 0.5 * infection_loss + 0.5 * organ_loss

        # Binary predictions
        infection_preds = (infection_output > 0.5).int()
        organ_preds = (organ_output > 0.5).int()
        sepsis_preds = (sepsis_prob > 0.5).int()

        # Compute accuracy for each task
        infection_accuracy = (infection_preds == y_infection).float().mean()
        organ_accuracy = (organ_preds == y_organ).float().mean()
        sepsis_accuracy = (sepsis_preds == y_sepsis).float().mean()

        # Log accuracies
        self.log('tr_inf_acc', infection_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('tr_org_acc', organ_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('tr_sep_acc', sepsis_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        # Metrics for Infection
        infection_precision, infection_recall, infection_f1 = self.calculate_safe_metrics(y_infection, infection_preds)
        infection_auroc = self.calculate_auroc_safe(y_infection, infection_output)
        self.log('tr_inf_prec', infection_precision, prog_bar=False)
        self.log('tr_inf_rec', infection_recall, prog_bar=False)
        self.log('tr_inf_f1', infection_f1, prog_bar=False)
        if infection_auroc is not None:
            self.log('tr_inf_auroc', infection_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Metrics/Tr_inf_AUROC', infection_auroc, self.global_step)

        # Metrics for Organ Dysfunction
        organ_precision, organ_recall, organ_f1 = self.calculate_safe_metrics(y_organ, organ_preds)
        organ_auroc = self.calculate_auroc_safe(y_organ, organ_output)
        self.log('tr_org_prec', organ_precision, prog_bar=False)
        self.log('tr_org_rec', organ_recall, prog_bar=False)
        self.log('tr_org_f1', organ_f1, prog_bar=False)
        if organ_auroc is not None:
            self.log('tr_org_auroc', organ_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Metrics/Tr_Organ_AUROC', organ_auroc, self.global_step)

        # Metrics for Sepsis
        sepsis_precision, sepsis_recall, sepsis_f1 = self.calculate_safe_metrics(y_sepsis, sepsis_preds)
        sepsis_auroc = self.calculate_auroc_safe(y_sepsis, sepsis_prob)
        self.log('tr_sep_prec', sepsis_precision, prog_bar=False)
        self.log('tr_sep_rec', sepsis_recall, prog_bar=False)
        self.log('tr_sep_f1', sepsis_f1, prog_bar=False)
        if sepsis_auroc is not None:
            self.log('tr_sep_auroc', sepsis_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Metrics/Tr_Sepsis_AUROC', sepsis_auroc, self.global_step)

        # Log total loss
        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.logger.experiment.add_scalar('Metrics/Training_Loss', loss, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_infection, y_organ, y_sepsis = batch
        infection_output, organ_output = self(x)
        sepsis_prob = infection_output.squeeze() * organ_output.squeeze()

        # Compute losses
        infection_loss = self.compute_loss(infection_output, y_infection, self.infection_class_weights.to(self.device))
        organ_loss = self.compute_loss(organ_output, y_organ, self.organ_class_weights.to(self.device))
        loss = 0.5 * infection_loss + 0.5 * organ_loss

        # Binary predictions
        infection_preds = (infection_output > 0.5).int()
        organ_preds = (organ_output > 0.5).int()
        sepsis_preds = (sepsis_prob > 0.5).int()

        # Compute accuracy for each task
        infection_accuracy = (infection_preds == y_infection).float().mean()
        organ_accuracy = (organ_preds == y_organ).float().mean()
        sepsis_accuracy = (sepsis_preds == y_sepsis).float().mean()

        # Log accuracies
        self.log('val_inf_acc', infection_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_org_acc', organ_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sep_acc', sepsis_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Metrics for Infection
        infection_precision, infection_recall, infection_f1 = self.calculate_safe_metrics(y_infection, infection_preds)
        infection_auroc = self.calculate_auroc_safe(y_infection, infection_output)
        self.log('val_inf_prec', infection_precision, prog_bar=False)
        self.log('val_inf_rec', infection_recall, prog_bar=False)
        self.log('val_inf_f1', infection_f1, prog_bar=False)
        if infection_auroc is not None:
            self.log('val_inf_auroc', infection_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Validation/Inf_AUROC', infection_auroc, self.global_step)
            self.logger.experiment.add_scalar('Validation/Inf_F1', infection_f1, self.global_step)

        # Metrics for Organ Dysfunction
        organ_precision, organ_recall, organ_f1 = self.calculate_safe_metrics(y_organ, organ_preds)
        organ_auroc = self.calculate_auroc_safe(y_organ, organ_output)
        self.log('val_org_prec', organ_precision, prog_bar=False)
        self.log('val_org_rec', organ_recall, prog_bar=False)
        self.log('val_org_f1', organ_f1, prog_bar=False)
        if organ_auroc is not None:
            self.log('val_org_auroc', organ_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Validation/Org_AUROC', organ_auroc, self.global_step)
            self.logger.experiment.add_scalar('Validation/Org_F1', organ_f1, self.global_step)

        # Metrics for Sepsis
        sepsis_precision, sepsis_recall, sepsis_f1 = self.calculate_safe_metrics(y_sepsis, sepsis_preds)
        sepsis_auroc = self.calculate_auroc_safe(y_sepsis, sepsis_prob)
        self.log('val_sep_prec', sepsis_precision, prog_bar=False)
        self.log('val_sep_rec', sepsis_recall, prog_bar=False)
        self.log('val_sep_f1', sepsis_f1, prog_bar=False)
        if sepsis_auroc is not None:
            self.log('val_sep_auroc', sepsis_auroc, prog_bar=True)
            self.logger.experiment.add_scalar('Validation/Sepsis_AUROC', sepsis_auroc, self.global_step)
            self.logger.experiment.add_scalar('Validation/Sepsis_F1', sepsis_f1, self.global_step)

        # Log total validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.add_scalar('Validation/Loss', loss, self.global_step)

        return {'val_loss': loss, 'sepsis_prob': sepsis_prob}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
