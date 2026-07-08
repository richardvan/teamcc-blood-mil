import time                                    ## NEW CODE
import torch
import numpy as np
import glob, os, shutil, zipfile             ## OLD CODE
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
# from google.colab import drive                 ## OLD CODE

# drive.mount('/content/drive')                 ## OLD CODE

# DATA_DIR = '/content/data'                     ## OLD CODE
# SAVE_DIR = '/content/drive/MyDrive/SVM/'        ## OLD CODE

# shutil.copy('/content/drive/MyDrive/data.zip', '/content/data.zip')   ## OLD CODE
# with zipfile.ZipFile('/content/data.zip', 'r') as z:                  ## OLD CODE
#   z.extractall('/content/data')                                      ## OLD CODE
# print("complete")                                                    ## OLD CODE

# image_paths = glob.glob(DATA_DIR + '/**/*.tif', recursive=True)       ## OLD CODE
# print(len(image_paths))                                              ## OLD CODE

DATA_DIR = '/home/sp00001/blood_mil_project/organized_data'          ## NEW CODE
SAVE_DIR = '/home/sp00001/blood_mil_project/richard_scripts/SVM_binary/'    ## NEW CODE
METADATA_PATH = '/home/sp00001/blood_mil_project/metadata_for_binary.csv'                  ## NEW CODE

meta_df = pd.read_csv(METADATA_PATH)                                 ## NEW CODE

os.makedirs(SAVE_DIR, exist_ok=True)                                       ## NEW CODE
X_PATH = os.path.join(SAVE_DIR, 'X.npy')                                   ## NEW CODE
GROUPS_PATH = os.path.join(SAVE_DIR, 'groups.npy')                         ## NEW CODE

if os.path.exists(X_PATH) and os.path.exists(GROUPS_PATH):                 ## NEW CODE
  print("found existing features, loading", X_PATH, "and", GROUPS_PATH)    ## NEW CODE
  X = np.load(X_PATH)                                                      ## NEW CODE
  groups = np.load(GROUPS_PATH)                                            ## NEW CODE
else:                                                                       ## NEW CODE
  feature_extraction_start = time.time()                                   ## NEW CODE
  image_paths = []                                                     ## NEW CODE
  for folder_name in meta_df['folder']:                                ## NEW CODE
    image_paths.extend(glob.glob(os.path.join(DATA_DIR, folder_name, '*.tif')))  ## NEW CODE
  print(len(image_paths))                                              ## NEW CODE

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

  np.save(X_PATH, X)                                                       ## NEW CODE
  np.save(GROUPS_PATH, groups)                                             ## NEW CODE

  feature_extraction_end = time.time()                                     ## NEW CODE
  print("feature extraction took", feature_extraction_end - feature_extraction_start, "seconds")  ## NEW CODE

# df = pd.read_excel(SAVE_DIR + 'metadata.xlsx', sheet_name='metadata')       ## OLD CODE
# df['label'] = (df['bag_label'] != 'control').astype(int)                    ## OLD CODE
# patient_label = dict(zip(df['patient_id'], df['label']))                    ## OLD CODE

df = meta_df.copy()                                                     ## NEW CODE
df['label'] = df['status'].astype(int)                                  ## NEW CODE (metadata_for_binary.csv already encodes status as 0=normal/1=cancer)
patient_label = dict(zip(df['folder'], df['label']))                    ## NEW CODE
HOLDOUT_PATH = '/home/sp00001/blood_mil_project/holdout_data/holdout_patients.txt'  ## NEW CODE
with open(HOLDOUT_PATH) as f:                                                        ## NEW CODE
  holdout_patients = set(line.strip() for line in f if line.strip())                ## NEW CODE

CV_SPLITS_DIR = '/home/sp00001/blood_mil_project/cv_splits'          ## NEW CODE
fold_covered_patients = set()                                        ## NEW CODE
for fold_name in sorted(os.listdir(CV_SPLITS_DIR)):                  ## NEW CODE
  fold_dir = os.path.join(CV_SPLITS_DIR, fold_name)                  ## NEW CODE
  with open(os.path.join(fold_dir, 'test_patients.txt')) as f:       ## NEW CODE
    fold_covered_patients.update(line.strip() for line in f if line.strip())  ## NEW CODE

patients = np.unique(groups)
patients = np.array([p for p in patients                                    ## NEW CODE
                      if p not in holdout_patients and p in fold_covered_patients])  ## NEW CODE

X_patient = []
y_patient = []

for pid in patients:
  mask = groups == pid
  avg_vector = X[mask].mean(axis=0)
  X_patient.append(avg_vector)
  y_patient.append(patient_label[pid])

X_patient = np.array(X_patient)
y_patient = np.array(y_patient)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone                                            ## NEW CODE
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict  ## NEW CODE (added cross_val_predict)
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report

# X_train, X_test, y_train, y_test = train_test_split(                     ## OLD CODE
#     X_patient, y_patient, test_size=0.2, stratify=y_patient, random_state=42)  ## OLD CODE

X_train = X_patient                                                    ## NEW CODE
y_train = y_patient                                                     ## NEW CODE

patient_to_idx = {pid: i for i, pid in enumerate(patients)}           ## NEW CODE

custom_cv = []                                                        ## NEW CODE
for fold_name in sorted(os.listdir(CV_SPLITS_DIR)):                   ## NEW CODE
  fold_dir = os.path.join(CV_SPLITS_DIR, fold_name)                   ## NEW CODE
  with open(os.path.join(fold_dir, 'train_patients.txt')) as f:       ## NEW CODE
    train_ids = [line.strip() for line in f if line.strip()]          ## NEW CODE
  with open(os.path.join(fold_dir, 'test_patients.txt')) as f:        ## NEW CODE
    test_ids = [line.strip() for line in f if line.strip()]           ## NEW CODE
  train_idx = np.array([patient_to_idx[pid] for pid in train_ids])    ## NEW CODE
  test_idx = np.array([patient_to_idx[pid] for pid in test_ids])      ## NEW CODE
  custom_cv.append((train_idx, test_idx))                             ## NEW CODE

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced")),
])

param_grid = {
    "svm__kernel" : ["linear", "rbf"],
    "svm__C" : [0.1, 1, 10],
    "svm__gamma" : ["scale", 0.01, 0.001],
}

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)         ## OLD CODE
grid = GridSearchCV(pipe, param_grid, cv=custom_cv,                       ## NEW CODE (cv=custom_cv instead of cv=cv)
                    scoring="balanced_accuracy", n_jobs=-1)

grid.fit(X_train, y_train)

import joblib                                                             ## NEW CODE
MODEL_PATH = os.path.join(SAVE_DIR, 'svm_best_model.joblib')              ## NEW CODE
joblib.dump(grid.best_estimator_, MODEL_PATH)                             ## NEW CODE
print("saved model to", MODEL_PATH)                                       ## NEW CODE

# pred = grid.predict(X_test)                                             ## OLD CODE
pred = cross_val_predict(grid.best_estimator_, X_train, y_train, cv=custom_cv)  ## NEW CODE

print("best parameter:", grid.best_params_)
print("best CV balanced accuracy:", grid.best_score_)                     ## NEW CODE
# print("balanced accuracy:", balanced_accuracy_score(y_test, pred))       ## OLD CODE
print("balanced accuracy:", balanced_accuracy_score(y_train, pred))       ## NEW CODE
# print("F1 score:", f1_score(y_test, pred, average="macro"))              ## OLD CODE
print("F1 score:", f1_score(y_train, pred, average="macro"))              ## NEW CODE
# print(classification_report(y_test, pred))                               ## OLD CODE
print(classification_report(y_train, pred))                               ## NEW CODE
# print(confusion_matrix(y_test, pred))                                    ## OLD CODE
print(confusion_matrix(y_train, pred))                                    ## NEW CODE

REPORT_PATH = os.path.join(SAVE_DIR, 'classification_report.txt')         ## NEW CODE
with open(REPORT_PATH, 'w') as f:                                          ## NEW CODE
  f.write("=== Grid search results (mean balanced accuracy across folds) ===\n")  ## NEW CODE
  results_order = np.argsort(-grid.cv_results_['mean_test_score'])        ## NEW CODE
  for i in results_order:                                                 ## NEW CODE
    f.write(f"{grid.cv_results_['params'][i]}: {grid.cv_results_['mean_test_score'][i]:.4f}\n")  ## NEW CODE
  f.write(f"\nBest params: {grid.best_params_} (mean balanced accuracy: {grid.best_score_:.4f})\n\n")  ## NEW CODE

  for fold_i, (train_idx, test_idx) in enumerate(custom_cv, start=1):      ## NEW CODE
    fold_model = clone(pipe).set_params(**grid.best_params_)              ## NEW CODE
    fold_model.fit(X_train[train_idx], y_train[train_idx])                ## NEW CODE
    fold_pred = fold_model.predict(X_train[test_idx])                     ## NEW CODE
    f.write(f"=== Fold {fold_i} ===\n")                                   ## NEW CODE
    f.write(classification_report(y_train[test_idx], fold_pred))          ## NEW CODE
    f.write("\n")                                                         ## NEW CODE
  f.write("=== Aggregate (all folds combined) ===\n")                     ## NEW CODE
  f.write(classification_report(y_train, pred))                           ## NEW CODE
print("saved classification report to", REPORT_PATH)                      ## NEW CODE

CONF_MATRIX_PATH = os.path.join(SAVE_DIR, 'confusion_matrix.txt')         ## NEW CODE
np.savetxt(CONF_MATRIX_PATH, confusion_matrix(y_train, pred), fmt='%d')  ## NEW CODE
print("saved confusion matrix to", CONF_MATRIX_PATH)                      ## NEW CODE





import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

def quick_eval(Xp, yp, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        Xp, yp, test_size=0.2, stratify=yp, random_state=seed)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("svm", SVC(kernel="rbf", C=1, class_weight="balanced"))])
    pipe.fit(Xtr, ytr)
    return balanced_accuracy_score(yte, pipe.predict(Xte))

# 1) Robustness across seeds — is 1.0 a lucky single split?
scores = [quick_eval(X_patient, y_patient, s) for s in range(10)]
print("real labels :", np.round(scores, 3), "mean", np.mean(scores).round(3))

# 2) Label-shuffle test — if shuffled labels still score high, something leaks
shuf = [quick_eval(X_patient, np.random.permutation(y_patient), s) for s in range(10)]
print("shuffled    :", np.round(shuf, 3), "mean", np.mean(shuf).round(3))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X_patient_scaled = StandardScaler().fit_transform(X_patient)
pca = PCA(n_components=2).fit(X_patient_scaled)
Z_pca = pca.transform(X_patient_scaled)
plt.scatter(Z_pca[y_patient==0,0], Z_pca[y_patient==0,1], label="control", alpha=.6)
plt.scatter(Z_pca[y_patient==1,0], Z_pca[y_patient==1,1], label="AML", alpha=.6)
plt.legend(); plt.title("Patient embeddings (PCA)")
var_pct = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({var_pct[0]:.0f}%)")
plt.ylabel(f"PC2 ({var_pct[1]:.0f}%)")
PCA_PLOT_PATH = os.path.join(SAVE_DIR, 'pca_plot.png')                    ## NEW CODE
plt.savefig(PCA_PLOT_PATH)                                                 ## NEW CODE
print("saved PCA plot to", PCA_PLOT_PATH)                                  ## NEW CODE
plt.show()




