import time
import torch
import numpy as np
import glob, os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import joblib
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report

DATA_DIR = '/home/sp00001/blood_mil_project/organized_data'
SAVE_DIR = '/home/sp00001/blood_mil_project/richard_scripts/SVM_multiclass/'
METADATA_PATH = '/home/sp00001/blood_mil_project/metadata_for_multiclass.csv'
HOLDOUT_PATH = '/home/sp00001/blood_mil_project/holdout_data_for_multiclass/holdout_patients.txt'
MODEL_PATH = os.path.join(SAVE_DIR, 'svm_best_model.joblib')

# holdout_data_for_multiclass/holdout_patients.txt lists the folder values that
# j_SVM_multiclass.py excluded from training/CV (via holdout_patients + fold_covered_patients
# filters) so this set stays truly unseen. This script is the first place that set gets
# loaded/evaluated.

meta_df = pd.read_csv(METADATA_PATH)

with open(HOLDOUT_PATH) as f:
  holdout_patients = set(line.strip() for line in f if line.strip())

label_categories = sorted(meta_df['status'].unique())                  ## label encoding must match j_SVM_multiclass.py, computed from the full metadata
label_to_int = {label: i for i, label in enumerate(label_categories)}

meta_df = meta_df[meta_df['folder'].isin(holdout_patients)]
print(len(meta_df), "holdout patients found in metadata")

os.makedirs(SAVE_DIR, exist_ok=True)
X_PATH = os.path.join(SAVE_DIR, 'X_holdout.npy')
GROUPS_PATH = os.path.join(SAVE_DIR, 'groups_holdout.npy')

if os.path.exists(X_PATH) and os.path.exists(GROUPS_PATH):
  print("found existing features, loading", X_PATH, "and", GROUPS_PATH)
  X = np.load(X_PATH)
  groups = np.load(GROUPS_PATH)
else:
  feature_extraction_start = time.time()
  image_paths = []
  for folder_name in meta_df['folder']:
    image_paths.extend(glob.glob(os.path.join(DATA_DIR, folder_name, '*.tif')))
  print(len(image_paths))

  device = "cuda" if torch.cuda.is_available() else "cpu"

  patient_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]

  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  class CellDataset(Dataset):
    def __init__(self, paths, transform):
      self.paths = paths
      self.transform = transform

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, i):
      img = Image.open(self.paths[i]).convert("RGB")
      return self.transform(img), i

  encoder = models.resnet50(weights="IMAGENET1K_V2")
  encoder.fc = torch.nn.Identity()
  encoder.eval()
  for p in encoder.parameters():
    p.requires_grad = False
  encoder.to(device)

  dataset = CellDataset(image_paths, transform)
  loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

  features = [None]*len(dataset)

  with torch.no_grad():
    for imgs, idx in loader:
      out = encoder(imgs.to(device))
      out = out.cpu().numpy()
      for vector, i in zip(out, idx.numpy()):
        features[i] = vector

  X = np.stack(features)
  groups = np.array(patient_ids)

  np.save(X_PATH, X)
  np.save(GROUPS_PATH, groups)

  feature_extraction_end = time.time()
  print("feature extraction took", feature_extraction_end - feature_extraction_start, "seconds")

df = meta_df.copy()
df['label'] = df['status'].map(label_to_int)
patient_label = dict(zip(df['folder'], df['label']))

patients = np.unique(groups)

X_patient = []
y_patient = []

for pid in patients:
  mask = groups == pid
  avg_vector = X[mask].mean(axis=0)
  X_patient.append(avg_vector)
  y_patient.append(patient_label[pid])

X_patient = np.array(X_patient)
y_patient = np.array(y_patient)

pipe = joblib.load(MODEL_PATH)
print("loaded model from", MODEL_PATH)

pred = pipe.predict(X_patient)

print("holdout balanced accuracy:", balanced_accuracy_score(y_patient, pred))
print("holdout F1 score:", f1_score(y_patient, pred, average="macro"))
print(classification_report(y_patient, pred, target_names=label_categories))
print(confusion_matrix(y_patient, pred))

REPORT_PATH = os.path.join(SAVE_DIR, 'holdout_classification_report.txt')
with open(REPORT_PATH, 'w') as f:
  f.write(classification_report(y_patient, pred, target_names=label_categories))
print("saved holdout classification report to", REPORT_PATH)

CONF_MATRIX_PATH = os.path.join(SAVE_DIR, 'holdout_confusion_matrix.txt')
np.savetxt(CONF_MATRIX_PATH, confusion_matrix(y_patient, pred), fmt='%d')
print("saved holdout confusion matrix to", CONF_MATRIX_PATH)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fit the PCA/scaler on the training patients (same population the model was trained
# on) so the holdout points land on the same axes as the training PCA plot, rather
# than fitting a fresh (noisier) PCA on just the holdout points.
TRAIN_X_PATH = os.path.join(SAVE_DIR, 'X.npy')
TRAIN_GROUPS_PATH = os.path.join(SAVE_DIR, 'groups.npy')
full_meta_df = pd.read_csv(METADATA_PATH)
full_meta_df['label'] = full_meta_df['status'].map(label_to_int)
full_patient_label = dict(zip(full_meta_df['folder'], full_meta_df['label']))

X_train_all = np.load(TRAIN_X_PATH)
groups_train_all = np.load(TRAIN_GROUPS_PATH)
train_patients = np.array([p for p in np.unique(groups_train_all) if p not in holdout_patients])

X_train_patient = []
y_train_patient = []
for pid in train_patients:
  mask = groups_train_all == pid
  X_train_patient.append(X_train_all[mask].mean(axis=0))
  y_train_patient.append(full_patient_label[pid])
X_train_patient = np.array(X_train_patient)
y_train_patient = np.array(y_train_patient)

scaler = StandardScaler().fit(X_train_patient)
pca = PCA(n_components=2).fit(scaler.transform(X_train_patient))

Z_train = pca.transform(scaler.transform(X_train_patient))
Z_holdout = pca.transform(scaler.transform(X_patient))

colors = plt.cm.tab10(np.linspace(0, 1, len(label_categories)))       ## distinct color per class, since class count is no longer fixed at 2
ax = plt.gca()
train_handles = []
holdout_handles = []
for class_idx, class_name in enumerate(label_categories):
  train_handle = ax.scatter(Z_train[y_train_patient==class_idx,0], Z_train[y_train_patient==class_idx,1],
                             label=class_name, alpha=.6, c=[colors[class_idx]])
  holdout_handle = ax.scatter(Z_holdout[y_patient==class_idx,0], Z_holdout[y_patient==class_idx,1],
                               label=class_name, marker="x", s=80, alpha=1, c=[colors[class_idx]])
  train_handles.append(train_handle)
  holdout_handles.append(holdout_handle)

train_legend = ax.legend(train_handles, label_categories, title="train",
                          loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
ax.add_artist(train_legend)
ax.legend(holdout_handles, label_categories, title="holdout",
          loc='lower left', bbox_to_anchor=(1.0, 0.0), fontsize=8)
var_pct = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({var_pct[0]:.0f}%)")
plt.ylabel(f"PC2 ({var_pct[1]:.0f}%)")
plt.title("Holdout patient embeddings (PCA, fit on training set)")
PCA_PLOT_PATH = os.path.join(SAVE_DIR, 'holdout_pca_plot.png')
plt.savefig(PCA_PLOT_PATH, bbox_inches='tight')
print("saved holdout PCA plot to", PCA_PLOT_PATH)
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

scores = pipe.predict_proba(X_patient)
y_bin = label_binarize(y_patient, classes=range(len(label_categories)))

plt.figure()
for class_idx, class_name in enumerate(label_categories):
  fpr, tpr, _ = roc_curve(y_bin[:, class_idx], scores[:, class_idx])
  class_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, color=colors[class_idx], linewidth=2,
           label=f"{class_name} (AUC={class_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (holdout)")
plt.legend(fontsize=8)
ROC_PLOT_PATH = os.path.join(SAVE_DIR, 'holdout_roc_curve_plot.png')
plt.savefig(ROC_PLOT_PATH, bbox_inches="tight")
print("saved holdout ROC curve plot to", ROC_PLOT_PATH)
plt.show()

N_CLASSES = len(label_categories)
cm = confusion_matrix(y_patient, pred, labels=range(N_CLASSES))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(label_categories, rotation=45, ha='right')
ax.set_yticklabels(label_categories)
ax.set_xlabel('predicted'); ax.set_ylabel('true')

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})', ha='center', va='center',
                color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=9)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Confusion Matrix (Holdout)')
plt.tight_layout()
CM_PLOT_PATH = os.path.join(SAVE_DIR, 'holdout_confusion_matrix_heatmap.png')
plt.savefig(CM_PLOT_PATH, bbox_inches='tight')
print("saved holdout confusion matrix heatmap to", CM_PLOT_PATH)
plt.show()


