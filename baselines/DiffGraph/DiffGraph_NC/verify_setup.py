"""
Verify that the setup is correct for GRAVEL binary classification
"""
import sys
sys.path.append('../..')
from DataHandler import DataHandler
from params import args
from Model import HGDM
import torch as t
from Utils.Utils import evaluate
import numpy as np

print("="*70)
print(" GRAVEL Binary Classification Setup Verification")
print("="*70)

# Test on pdns dataset
args.data = 'pdns'
args.epoch = 1
args.batch = 256
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

print("\n[Step 1/3] Loading data...")
handler = DataHandler()
handler.LoadData()

print(f"✓ Data loaded!")
print(f"  - Features: {handler.feature_list.shape}")
print(f"  - Labels: {handler.labels.shape} → Binary classification (2 classes)")
print(f"  - Train: {len(handler.train_idx[0])}, Val: {len(handler.val_idx[0])}, Test: {len(handler.test_idx[0])}")

print("\n[Step 2/3] Checking label distribution...")
labels = handler.labels
train_labels = labels[handler.train_idx[0]]
test_labels = labels[handler.test_idx[0]]

train_class_0 = t.sum(train_labels[:, 0] == 1).item()
train_class_1 = t.sum(train_labels[:, 1] == 1).item()
test_class_0 = t.sum(test_labels[:, 0] == 1).item()
test_class_1 = t.sum(test_labels[:, 1] == 1).item()

print(f"✓ Label distribution:")
print(f"  - Train: Class 0 (benign) = {train_class_0}, Class 1 (malicious) = {train_class_1}")
print(f"  - Test:  Class 0 (benign) = {test_class_0}, Class 1 (malicious) = {test_class_1}")

# Check test labels for AUC calculation
test_lbls = t.argmax(test_labels, dim=-1)
unique_test_classes = np.unique(test_lbls.cpu().numpy())
print(f"  - Unique classes in test set: {unique_test_classes}")

if len(unique_test_classes) < 2:
    print(f"  ⚠ WARNING: Test set only contains {len(unique_test_classes)} class(es)!")
    print(f"    This may cause issues with AUC calculation.")
else:
    print(f"  ✓ Test set contains both classes - AUC can be calculated")

print("\n[Step 3/3] Testing model forward pass...")
dim = handler.feature_list.shape[1]
nbclasses = labels.shape[1]
print(f"  - Input dim: {dim}, Output classes: {nbclasses}")

model = HGDM(dim).to(device)
print(f"✓ Model initialized")

# Test forward pass with a small batch
batch_indices = handler.train_idx[0][:64]
ancs = t.LongTensor(batch_indices.numpy())

try:
    nll_loss, diffloss = model.cal_loss(ancs, labels, handler.he_adjs, handler.feature_list)
    loss = nll_loss + diffloss
    print(f"✓ Forward pass successful: loss = {loss.item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test inference
try:
    with t.no_grad():
        embeds, scores = model.get_allembeds(handler.he_adjs, handler.feature_list)
    print(f"✓ Inference successful: embeddings shape = {embeds.shape}")
except Exception as e:
    print(f"✗ Inference failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print(" ✓ All checks passed! Ready to train.")
print("="*70)
print("\nRun training with:")
print(f"  python main.py --data pdns --epoch 100 --batch 256 --steps 200 --lr 3e-3")

