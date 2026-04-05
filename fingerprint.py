# ============================================
# Fingerprint Pattern Classification Project
# ============================================
# This script trains a CNN (ResNet18) model to classify
# fingerprint images into:
#   - Arch
#   - Loop
#   - Whorl
# ============================================

# Install required libraries (Colab only)
!pip install torch torchvision kaggle

# ============================================
# 📁 Mount Google Drive (Colab environment)
# ============================================
from google.colab import drive
drive.mount("/content/drive")

# Path to dataset
data_set = "/content/drive/MyDrive/Fingerprint_project"

# ============================================
# 📦 Import Libraries
# ============================================
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

# ============================================
# 🧠 Data Preprocessing
# ============================================
# Train transform includes augmentation to improve generalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # simulate variation
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet normalization
                         [0.229, 0.224, 0.225])
])

# Validation/Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================
# 📂 Load Dataset
# ============================================
train_data = datasets.ImageFolder(data_set + "/NISTDB4_RAW/train_set", transform=train_transform)
val_data   = datasets.ImageFolder(data_set + "/NISTDB4_RAW/val_set", transform=test_transform)
test_data  = datasets.ImageFolder(data_set + "/NISTDB4_RAW/test_set", transform=test_transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

# Print dataset info
print("Classes:", train_data.classes)
print("Train size:", len(train_data))
print("Val size:", len(val_data))
print("Test size:", len(test_data))

# ============================================
# 🧠 Model Setup (Transfer Learning)
# ============================================
# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers (feature extractor)
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ============================================
# ⚙️ Loss Function & Optimizer
# ============================================
# Weighted loss to reduce confusion (loop vs whorl)
weights = torch.tensor([1.0, 1.2, 1.2]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Train only final layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# ============================================
# 🔥 Training Loop
# ============================================
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # ---- Training Phase ----
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation Phase ----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")

# ============================================
# 📊 Test Evaluation
# ============================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", correct / total)

# ============================================
# 💾 Save Model
# ============================================
torch.save(model.state_dict(), "model_best_78.pth")

# ============================================
# 📊 Confusion Matrix (Model Analysis)
# ============================================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=train_data.classes)
disp.plot()
plt.show()