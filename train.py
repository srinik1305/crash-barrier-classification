# ML Model with ML flow and Model Registry 

import os
import random
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from mlflow.tracking import MlflowClient

BASE_DIR = r"C:\Users\Admin\capstone_proj\project-2-at-2026-05-09-14-04-ad8b975c"

DATASET_DIR = os.path.join(BASE_DIR, "processed_dataset")

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}"
)

mlflow.set_experiment("CrashBarrier_Classification")

REGISTERED_MODEL_NAME = "CrashBarrier_ResNet18"

BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-4
IMG_SIZE = 224
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

random.seed(SEED)
torch.manual_seed(SEED)

good_dir = os.path.join(DATASET_DIR, "Good")
damaged_dir = os.path.join(DATASET_DIR, "Damaged")

good_images = [
    (os.path.join(good_dir, f), 0)
    for f in os.listdir(good_dir)
]

damaged_images = [
    (os.path.join(damaged_dir, f), 1)
    for f in os.listdir(damaged_dir)
]

all_data = good_images + damaged_images

train_data, temp_data = train_test_split(
    all_data,
    test_size=0.30,
    random_state=SEED,
    stratify=[x[1] for x in all_data]
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=SEED,
    stratify=[x[1] for x in temp_data]
)

print(f"Train: {len(train_data)}")
print(f"Val  : {len(val_data)}")
print(f"Test : {len(test_data)}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CrashBarrierDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        image_path, label = self.data[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = CrashBarrierDataset(
    train_data,
    train_transform
)

val_dataset = CrashBarrierDataset(
    val_data,
    test_transform
)

test_dataset = CrashBarrierDataset(
    test_data,
    test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 2)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LR
)

with mlflow.start_run():

    mlflow.log_param("model", "ResNet18")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("image_size", IMG_SIZE)
    mlflow.log_param("device", str(DEVICE))

    best_val_acc = 0.0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        train_preds = []
        train_labels = []

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(
            train_labels,
            train_preds
        )

        model.eval()

        val_loss = 0.0

        val_preds = []
        val_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(
            val_labels,
            val_preds
        )

        val_precision = precision_score(
            val_labels,
            val_preds
        )

        val_recall = recall_score(
            val_labels,
            val_preds
        )

        val_f1 = f1_score(
            val_labels,
            val_preds
        )

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        mlflow.log_metric(
            "train_loss",
            train_loss / len(train_loader),
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss / len(val_loader),
            step=epoch
        )

        mlflow.log_metric(
            "train_accuracy",
            train_acc,
            step=epoch
        )

        mlflow.log_metric(
            "val_accuracy",
            val_acc,
            step=epoch
        )

        mlflow.log_metric(
            "val_precision",
            val_precision,
            step=epoch
        )

        mlflow.log_metric(
            "val_recall",
            val_recall,
            step=epoch
        )

        mlflow.log_metric(
            "val_f1",
            val_f1,
            step=epoch
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            model_path = os.path.join(
                BASE_DIR,
                "best_resnet18.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

    model.load_state_dict(
        torch.load(model_path)
    )

    model.eval()

    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    test_acc = accuracy_score(
        test_labels,
        test_preds
    )

    test_precision = precision_score(
        test_labels,
        test_preds
    )

    test_recall = recall_score(
        test_labels,
        test_preds
    )

    test_f1 = f1_score(
        test_labels,
        test_preds
    )

    test_auc = roc_auc_score(
        test_labels,
        test_probs
    )

    test_ap = average_precision_score(
        test_labels,
        test_probs
    )

    balanced_acc = balanced_accuracy_score(
        test_labels,
        test_preds
    )

    mcc = matthews_corrcoef(
        test_labels,
        test_preds
    )

    kappa = cohen_kappa_score(
        test_labels,
        test_preds
    )

    cm = confusion_matrix(
        test_labels,
        test_preds
    )

    report = classification_report(
        test_labels,
        test_preds,
        target_names=["Good", "Damaged"]
    )

    print("\nTest Results")
    print(f"Accuracy         : {test_acc:.4f}")
    print(f"Precision        : {test_precision:.4f}")
    print(f"Recall           : {test_recall:.4f}")
    print(f"F1 Score         : {test_f1:.4f}")
    print(f"ROC AUC          : {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"MCC              : {mcc:.4f}")
    print(f"Cohen Kappa      : {kappa:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_average_precision", test_ap)
    mlflow.log_metric("balanced_accuracy", balanced_acc)
    mlflow.log_metric("mcc", mcc)
    mlflow.log_metric("cohen_kappa", kappa)

    report_path = os.path.join(
        BASE_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    cm_path = os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Good", "Damaged"],
        yticklabels=["Good", "Damaged"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(cm_path)

    plt.close()

    mlflow.log_artifact(cm_path)

    roc_path = os.path.join(
        BASE_DIR,
        "roc_curve.png"
    )

    fpr, tpr, _ = roc_curve(
        test_labels,
        test_probs
    )

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.tight_layout()

    plt.savefig(roc_path)

    plt.close()

    mlflow.log_artifact(roc_path)

    pr_path = os.path.join(
        BASE_DIR,
        "precision_recall_curve.png"
    )

    precision_vals, recall_vals, _ = precision_recall_curve(
        test_labels,
        test_probs
    )

    plt.figure(figsize=(6, 5))

    plt.plot(recall_vals, precision_vals)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(pr_path)

    plt.close()

    mlflow.log_artifact(pr_path)

    history_df = pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history
    })

    history_csv = os.path.join(
        BASE_DIR,
        "training_history.csv"
    )

    history_df.to_csv(history_csv, index=False)

    mlflow.log_artifact(history_csv)

    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="resnet18_model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    mlflow.log_artifact(model_path)

    print("\nModel registered successfully")

    print(f"Registered Model : {REGISTERED_MODEL_NAME}")

    print(f"Model URI        : {model_info.model_uri}")

    client = MlflowClient()

    latest_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    print("\nRegistered Model Versions")

    for mv in latest_versions:

        print(
            f"Version: {mv.version} | "
            f"Stage: {mv.current_stage}"
        )

    PROMOTE_TO_STAGING = (
        test_f1 >= 0.85 and
        test_recall >= 0.90 and
        test_auc >= 0.90
    )

    if PROMOTE_TO_STAGING:

        latest_version = latest_versions[-1].version

        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            stage="Staging"
        )

        print("\nModel promoted to STAGING")

    else:

        print("\nModel did NOT meet promotion criteria")

print("\nTraining completed.")