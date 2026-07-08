import torch
import numpy as np
import glob, os, shutil, zipfile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from google.colab import drive
from sklearn.preprocessing import LabelEncoder

drive.mount('/content/drive')

DATA_DIR = '/content/data'
SAVE_DIR = '/content/drive/MyDrive/SVM/'

shutil.copy('/content/drive/MyDrive/data.zip', '/content/data.zip')
with zipfile.ZipFile('/content/data.zip', 'r') as z:
  z.extractall('/content/data')
print("complete")

image_paths = glob.glob(DATA_DIR + '/**/*.tif', recursive=True)
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

os.makedirs(SAVE_DIR, exist_ok=True)
np.save(SAVE_DIR + 'X.npy', X)
np.save(SAVE_DIR + 'groups.npy', groups)

X = np.load(SAVE_DIR + 'X.npy')
groups = np.load(SAVE_DIR + 'groups.npy')

df = pd.read_excel(SAVE_DIR + 'metadata.xlsx', sheet_name='metadata')
le = LabelEncoder()
df['label'] = le.fit_transform(df['bag_label'])

patient_label = dict(zip(df['patient_id'], df['label']))
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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X_patient, y_patient, test_size=0.2, stratify=y_patient, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced")),
])

param_grid = {
    "svm__kernel" : ["linear", "rbf"],
    "svm__C" : [0.1, 1, 10],
    "svm__gamma" : ["scale", 0.01, 0.001],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv,
                    scoring="balanced_accuracy", n_jobs=-1)

grid.fit(X_train, y_train)

pred = grid.predict(X_test)

print("best parameter:", grid.best_params_)
print("balanced accuracy:", balanced_accuracy_score(y_test, pred))
print("F1 score:", f1_score(y_test, pred, average="macro"))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))





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

Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_patient))
for cls in np.unique(y_patient):
    plt.scatter(Z[y_patient==cls, 0], Z[y_patient==cls, 1],
                label=le.inverse_transform([cls])[0], alpha=.6)
plt.legend(); plt.title("Patient embeddings (PCA, 5 subtypes)"); plt.show()


