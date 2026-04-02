import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from dataset.loader import get_dataloaders
from models.model import get_model


def train():

    # ================= DEVICE =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # ================= DATA =================
    DATA_PATH = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"

    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH)

    # ================= MODEL =================
    model = get_model().to(device)

    # ================= LOSS (Class Imbalance Fix) =================
    weights = torch.tensor([2.5, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ================= OPTIMIZER =================
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ================= TRAINING =================
    epochs = 5
    best_acc = 0.0

    for epoch in range(epochs):

        print(f"\n🔥 Epoch {epoch+1}/{epochs}")

        # -------- TRAIN --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        print(f"📉 Training Loss: {running_loss:.4f}")
        print(f"✅ Training Accuracy: {train_acc:.2f}%")

        # -------- VALIDATION --------
        model.eval()
        val_preds = []
        val_labels = []

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total

        print("\n📊 VALIDATION RESULTS:")
        print("Accuracy:", val_acc)
        print(confusion_matrix(val_labels, val_preds))
        print(classification_report(val_labels, val_preds))

        # -------- SAVE BEST MODEL --------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Best model saved!")

    # ================= TEST =================
    print("\n🔥 FINAL TEST RESULTS (Unseen Data):")

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print(confusion_matrix(test_labels, test_preds))
    print(classification_report(test_labels, test_preds))

    # ================= FINAL SAVE =================
    torch.save(model.state_dict(), "final_model.pth")
    print("\n✅ Final model saved as final_model.pth")


if __name__ == "__main__":
    train()
