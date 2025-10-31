import torch
from torchvision import transforms, datasets
from modules import convnext_tiny, convnext_small
# ===========================
#  Config
# ===========================
MODEL_PATH = r"E:\comp3710\demo3_git\demo3_git\outputs\exp1\best.pt"  
MODEL_TYPE = "tiny" 
DATA_DIR = r"E:\comp3710\demo3_git\demo3_git\test_data"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
#  Data transforms
# ===========================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===========================
#  Dataset & DataLoader
# ===========================
test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===========================
#  Build model
# ===========================
num_classes = len(test_dataset.classes)
if MODEL_TYPE == "tiny":
    model = convnext_tiny(num_classes=num_classes)
elif MODEL_TYPE == "small":
    model = convnext_small(num_classes=num_classes)
else:
    raise ValueError("MODEL_TYPE must be 'tiny' or 'small'")

# ===========================
#  Load checkpoint
# ===========================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()  # evaluation mode

print(f"Loaded model '{MODEL_PATH}' on device {DEVICE}")
print(f"Classes: {test_dataset.classes}")

# ===========================
#  Predict
# ===========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# ===========================
#  Print results
# ===========================
correct = sum([p == t for p, t in zip(all_preds, all_labels)])
total = len(all_labels)
accuracy = correct / total if total > 0 else 0.0

print(f"Test samples: {total}")
print(f"Accuracy: {accuracy:.4f}")

# Optionally, print per-sample prediction
for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
    print(f"Sample {i}: Predicted = {test_dataset.classes[pred]}, True = {test_dataset.classes[label]}")