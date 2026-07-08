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
SAVE_DIR = '/home/sp00001/blood_mil_project/richard_scripts/SVM_binary/'
METADATA_PATH = '/home/sp00001/blood_mil_project/metadata_for_binary.csv'
HOLDOUT_PATH = '/home/sp00001/blood_mil_project/holdout_data/holdout_patients.txt'
MODEL_PATH = os.path.join(SAVE_DIR, 'svm_best_model.joblib')

# holdout_data/holdout_patients.txt lists the folder_name values that j_SVM_binary.py
# excluded from training/CV (via holdout_patients + fold_covered_patients filters) so
# this set stays truly unseen. This script is the first place that set gets loaded/evaluated.

meta_df = pd.read_csv(METADATA_PATH)

with open(HOLDOUT_PATH) as f:
  holdout_patients = set(line.strip() for line in f if line.strip())

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
df['label'] = df['status'].astype(int)
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
print(classification_report(y_patient, pred))
print(confusion_matrix(y_patient, pred))

REPORT_PATH = os.path.join(SAVE_DIR, 'holdout_classification_report.txt')
with open(REPORT_PATH, 'w') as f:
  f.write(classification_report(y_patient, pred))
print("saved holdout classification report to", REPORT_PATH)

CONF_MATRIX_PATH = os.path.join(SAVE_DIR, 'holdout_confusion_matrix.txt')
np.savetxt(CONF_MATRIX_PATH, confusion_matrix(y_patient, pred), fmt='%d')
print("saved holdout confusion matrix to", CONF_MATRIX_PATH)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fit the PCA/scaler on the training patients (same population the model was trained
# on) so the holdout points land on the same axes as the training PCA plot, rather
# than fitting a fresh (noisier) PCA on just the 28 holdout points.
TRAIN_X_PATH = os.path.join(SAVE_DIR, 'X.npy')
TRAIN_GROUPS_PATH = os.path.join(SAVE_DIR, 'groups.npy')
full_meta_df = pd.read_csv(METADATA_PATH)
full_meta_df['label'] = full_meta_df['status'].astype(int)
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

plt.scatter(Z_train[y_train_patient==0,0], Z_train[y_train_patient==0,1], label="train: control", alpha=.3, c="tab:blue")
plt.scatter(Z_train[y_train_patient==1,0], Z_train[y_train_patient==1,1], label="train: AML", alpha=.3, c="tab:orange")
plt.scatter(Z_holdout[y_patient==0,0], Z_holdout[y_patient==0,1], label="holdout: control", marker="x", s=100, c="tab:blue")
plt.scatter(Z_holdout[y_patient==1,0], Z_holdout[y_patient==1,1], label="holdout: AML", marker="x", s=100, c="tab:orange")
plt.legend(); plt.title("Holdout patient embeddings (PCA, fit on training set)")
PCA_PLOT_PATH = os.path.join(SAVE_DIR, 'holdout_pca_plot.png')
plt.savefig(PCA_PLOT_PATH)
print("saved holdout PCA plot to", PCA_PLOT_PATH)
plt.show()
