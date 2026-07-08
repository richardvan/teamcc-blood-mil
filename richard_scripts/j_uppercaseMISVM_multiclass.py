#구글드라이브 연결, 압축해제

# from google.colab import drive                                          ## OLD CODE
# drive.mount('/content/drive')                                           ## OLD CODE

import os, zipfile

# SAVE_DIR = '/content/drive/MyDrive/SVM'                                 ## OLD CODE
# DATA_DIR = '/content'                                                   ## OLD CODE
# ZIP_PATH = '/content/drive/MyDrive/data.zip'                            ## OLD CODE

# expected_dir = os.path.join(DATA_DIR, 'CBFB_MYH11')                     ## OLD CODE

# if not os.path.exists(expected_dir):                                    ## OLD CODE
#   with zipfile.ZipFile(ZIP_PATH) as z:                                  ## OLD CODE
#     z.extractall('/content')                                           ## OLD CODE
#   print('압축 해제 완료')                                                ## OLD CODE

# else:                                                                   ## OLD CODE
#   print('이미 압축 해제됨')                                              ## OLD CODE

# assert os.path.isdir('/content/CBFB_MYH11/AQK'), '폴더가 없습니다: /content/CBFB_MYH11/AQK'  ## OLD CODE

# print(len([                                                             ## OLD CODE
#     f for f in os.listdir('/content/CBFB_MYH11/AQK')                    ## OLD CODE
#     if f.lower().endswith('.tif')                                       ## OLD CODE
# ]))                                                                      ## OLD CODE

import time                                                                 ## NEW CODE
import glob                                                                 ## NEW CODE
import numpy as np                                                         ## NEW CODE
import pandas as pd                                                        ## NEW CODE

DATA_DIR = '/home/sp00001/blood_mil_project/organized_data'                ## NEW CODE
SAVE_DIR = '/home/sp00001/blood_mil_project/richard_scripts/MIL_SVM_multiclass/'  ## NEW CODE
METADATA_PATH = '/home/sp00001/blood_mil_project/metadata_for_multiclass.csv'      ## NEW CODE

meta_df = pd.read_csv(METADATA_PATH)                                       ## NEW CODE
os.makedirs(SAVE_DIR, exist_ok=True)                                       ## NEW CODE

#설정값

import torch

# CLASSES = ['control', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']   ## OLD CODE
# CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}                        ## OLD CODE
# N_CLASSES = len(CLASSES)                                                    ## OLD CODE

CLASSES = sorted(meta_df['status'].unique())                                 ## NEW CODE
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}                         ## NEW CODE
N_CLASSES = len(CLASSES)                                                     ## NEW CODE

# FOLD = 1                                                                    ## OLD CODE
MODE = 'MI'

LR = 0.0001
WEIGHT_DECAY = 1e-2
EPOCHS = 40
SEED = 42

BATCH_IMG = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

#특징추출(ResNet50)

import torch.nn as nn
from torchvision import models, transforms

weights = models.ResNet50_Weights.IMAGENET1K_V2
backbone = models.resnet50(weights=weights)

backbone.fc = nn.Identity()
backbone.eval().to(device)
for p in backbone.parameters():
  p.requires_grad = False

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

FEATURE_DIM = 2048

#환자별 특징 추출, 캐시저장

from PIL import Image

# FEAT_CACHE = SAVE_DIR + '/features/'                                       ## OLD CODE
# os.makedirs(FEAT_CACHE, exist_ok=True)                                     ## OLD CODE

# def extract_one_patient(folder_name):                                      ## OLD CODE
#   parts = folder_name.split('.')                                          ## OLD CODE
#   folder = os.path.join(DATA_DIR, parts[1], parts[2])                     ## OLD CODE
#   files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.tif'))  ## OLD CODE
#   feats, batch = [], []                                                   ## OLD CODE
#   for f in files:                                                         ## OLD CODE
#     img = Image.open(os.path.join(folder, f)).convert('RGB')              ## OLD CODE
#     batch.append(tf(img))                                                 ## OLD CODE
#     if len(batch) == BATCH_IMG:                                           ## OLD CODE
#       x = torch.stack(batch).to(device)                                   ## OLD CODE
#       feats.append(backbone(x).cpu())                                     ## OLD CODE
#       batch = []                                                          ## OLD CODE

#   if batch:                                                               ## OLD CODE
#    x = torch.stack(batch).to(device)                                      ## OLD CODE
#    feats.append(backbone(x).cpu())                                        ## OLD CODE
#   return torch.cat(feats, dim=0)                                          ## OLD CODE

# #메타데이터 추출                                                           ## OLD CODE

# import pandas as pd                                                       ## OLD CODE

# meta = pd.read_csv(SAVE_DIR + '/metadata_server.csv')                     ## OLD CODE

# for _, row in meta.iterrows():                                            ## OLD CODE
#   pid = row['patient_id']                                                 ## OLD CODE
#   fpath = FEAT_CACHE + f'{pid}.pt'                                        ## OLD CODE
#   if os.path.exists(fpath):                                               ## OLD CODE
#     continue                                                              ## OLD CODE
#   feat = extract_one_patient(row['folder'])                               ## OLD CODE
#   torch.save(feat, fpath)                                                 ## OLD CODE
#   print(f'{pid}: {tuple(feat.shape)} 저장')                                ## OLD CODE

# print('특징 추출/캐시 완료')                                                ## OLD CODE

X_PATH = os.path.join(SAVE_DIR, 'X.npy')                                    ## NEW CODE
GROUPS_PATH = os.path.join(SAVE_DIR, 'groups.npy')                          ## NEW CODE

if os.path.exists(X_PATH) and os.path.exists(GROUPS_PATH):                  ## NEW CODE
  print("found existing features, loading", X_PATH, "and", GROUPS_PATH)     ## NEW CODE
  X = np.load(X_PATH)                                                       ## NEW CODE
  groups = np.load(GROUPS_PATH)                                             ## NEW CODE
else:                                                                        ## NEW CODE
  feature_extraction_start = time.time()                                    ## NEW CODE
  image_paths = []                                                          ## NEW CODE
  for folder_name in meta_df['folder']:                                     ## NEW CODE
    image_paths.extend(glob.glob(os.path.join(DATA_DIR, folder_name, '*.tif')))  ## NEW CODE
  print(len(image_paths))                                                   ## NEW CODE

  patient_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]  ## NEW CODE

  from torch.utils.data import Dataset, DataLoader                          ## NEW CODE

  class CellDataset(Dataset):                                                ## NEW CODE
    def __init__(self, paths, transform):                                   ## NEW CODE
      self.paths = paths                                                    ## NEW CODE
      self.transform = transform                                            ## NEW CODE

    def __len__(self):                                                      ## NEW CODE
      return len(self.paths)                                                ## NEW CODE

    def __getitem__(self, i):                                               ## NEW CODE
      img = Image.open(self.paths[i]).convert("RGB")                        ## NEW CODE
      return self.transform(img), i                                         ## NEW CODE

  dataset = CellDataset(image_paths, tf)                                    ## NEW CODE
  loader = DataLoader(dataset, batch_size=BATCH_IMG, shuffle=False, num_workers=2)  ## NEW CODE

  features = [None]*len(dataset)                                            ## NEW CODE

  with torch.no_grad():                                                     ## NEW CODE
    for imgs, idx in loader:                                                ## NEW CODE
      out = backbone(imgs.to(device))                                       ## NEW CODE
      out = out.cpu().numpy()                                               ## NEW CODE
      for vector, i in zip(out, idx.numpy()):                                ## NEW CODE
        features[i] = vector                                                ## NEW CODE

  X = np.stack(features)                                                    ## NEW CODE
  groups = np.array(patient_ids)                                            ## NEW CODE

  np.save(X_PATH, X)                                                        ## NEW CODE
  np.save(GROUPS_PATH, groups)                                              ## NEW CODE

  feature_extraction_end = time.time()                                     ## NEW CODE
  print("feature extraction took", feature_extraction_end - feature_extraction_start, "seconds")  ## NEW CODE

#라벨, 폴드

# def folder_to_label(folder):                                              ## OLD CODE
#   subtype = folder.split('.')[1]                                         ## OLD CODE
#   return CLASS_TO_IDX[subtype]                                           ## OLD CODE

# meta['label'] = meta['folder'].apply(folder_to_label)                    ## OLD CODE

# fold_col = f'fold_{FOLD}_status'                                         ## OLD CODE

# train_ids = meta[meta[fold_col] == 'train']['patient_id'].tolist()       ## OLD CODE
# val_ids = meta[meta[fold_col] == 'test']['patient_id'].tolist()          ## OLD CODE
# holdout_ids = meta[meta['is_holdout'] == True]['patient_id'].tolist()    ## OLD CODE

# print(f'train {len(train_ids)}명 / val {len(val_ids)}명 / holdout {len(holdout_ids)}명')  ## OLD CODE
# for k in range(1, 6):                                                     ## OLD CODE
#   fc = f'fold_{k}_status'                                                ## OLD CODE
#   n_train = (meta[fc] == 'train').sum()                                  ## OLD CODE
#   n_val = (meta[fc] == 'test').sum()                                     ## OLD CODE
#   dist = meta[meta[fc] == 'train']['label'].value_counts().sort_index().tolist()  ## OLD CODE
#   print(f'[fold {k}] train {n_train} / val {n_val} | {dist}')            ## OLD CODE

meta = meta_df.copy()                                                       ## NEW CODE
label_to_int = CLASS_TO_IDX                                                 ## NEW CODE
meta['label'] = meta['status'].map(label_to_int)                            ## NEW CODE
patient_label = dict(zip(meta['folder'], meta['label']))                    ## NEW CODE

HOLDOUT_PATH = '/home/sp00001/blood_mil_project/holdout_data_for_multiclass/holdout_patients.txt'  ## NEW CODE
with open(HOLDOUT_PATH) as f:                                                ## NEW CODE
  holdout_patients = set(line.strip() for line in f if line.strip())        ## NEW CODE

CV_SPLITS_DIR = '/home/sp00001/blood_mil_project/cv_splits_for_multiclass'   ## NEW CODE
fold_covered_patients = set()                                               ## NEW CODE
for fold_name in sorted(os.listdir(CV_SPLITS_DIR)):                         ## NEW CODE
  fold_dir = os.path.join(CV_SPLITS_DIR, fold_name)                         ## NEW CODE
  with open(os.path.join(fold_dir, 'test_patients.txt')) as f:              ## NEW CODE
    fold_covered_patients.update(line.strip() for line in f if line.strip())  ## NEW CODE

patients = np.unique(groups)                                                ## NEW CODE
patients = np.array([p for p in patients                                    ## NEW CODE
                      if p not in holdout_patients and p in fold_covered_patients])  ## NEW CODE

#bag 데이터 준비, 표준화

# def load_bags(id_list):                                                   ## OLD CODE
#   label_map = dict(zip(meta['patient_id'], meta['label']))               ## OLD CODE
#   bags = []                                                               ## OLD CODE
#   for pid in id_list:                                                     ## OLD CODE
#     feat = torch.load(FEAT_CACHE + f'{pid}.pt')                          ## OLD CODE
#     bags.append((feat, label_map[pid], pid))                             ## OLD CODE
#   return bags                                                             ## OLD CODE

# train_bags = load_bags(train_ids)                                        ## OLD CODE
# val_bags = load_bags(val_ids)                                            ## OLD CODE
# holdout_bags = load_bags(holdout_ids)                                    ## OLD CODE

# train_instances = torch.cat([f for f, _, _ in train_bags], dim=0)        ## OLD CODE
# feat_mean = train_instances.mean(dim=0)                                  ## OLD CODE
# feat_std = train_instances.std(dim=0) + 1e-6                             ## OLD CODE

def load_bags_from_groups(id_list):                                         ## NEW CODE
  bags = []                                                                 ## NEW CODE
  for pid in id_list:                                                       ## NEW CODE
    mask = groups == pid                                                    ## NEW CODE
    feat = torch.tensor(X[mask], dtype=torch.float32)                       ## NEW CODE
    bags.append((feat, patient_label[pid], pid))                            ## NEW CODE
  return bags                                                                ## NEW CODE

# def standardize(bags):                                                     ## OLD CODE (unused now; per-fold standardization is inlined in the 5-fold loop below)
#   return [((f-feat_mean) / feat_std, y, pid) for f, y, pid in bags]        ## OLD CODE

# train_bags = standardize(train_bags)                                      ## OLD CODE
# val_bags = standardize(val_bags)                                          ## OLD CODE
# holdout_bags = standardize(holdout_bags)                                  ## OLD CODE

# print('표준화 완료')                                                       ## OLD CODE

#MIL_SVM

torch.manual_seed(SEED)

class MIL_SVM(nn.Module):
  def __init__(self, in_dim, n_classes, mode='MI'):
    super().__init__()
    self.mode = mode
    self.classifier = nn.Linear(in_dim, n_classes)

  def forward(self, bag):
    if self.mode == 'MI':
      z = bag.mean(dim=0, keepdim=True)
      scores = self.classifier(z)
      return scores, None
    else:
      inst = self.classifier(bag)
      scores, _ = inst.max(dim=0, keepdim=True)
      return scores, inst


#학습 루프

def evaluate(model, bags, criterion):
  model.eval()
  correct = 0
  total_loss = 0.0
  with torch.no_grad():
    for feat, y, _ in bags:
      feat = feat.to(device)
      target = torch.tensor(y, device=device)

      scores, _ = model(feat)
      loss = criterion(scores, target)

      total_loss += loss.item()
      pred = scores.argmax(dim=1).item()
      correct += (pred == y)
  return correct / len(bags), total_loss / len(bags)

def train_model(mode, train_bags, val_bags):                                                          ## NEW CODE (dropped fold_col param)
  model = MIL_SVM(FEATURE_DIM, N_CLASSES, mode=mode).to(device)
  # counts = torch.tensor([ (meta[meta[fold_col]=='train']['label']==i).sum() for i in range(N_CLASSES)], dtype=torch.float)  ## OLD CODE
  counts = torch.tensor([sum(1 for _, y, _ in train_bags if y == i) for i in range(N_CLASSES)], dtype=torch.float)  ## NEW CODE
  w = (counts.sum() / (N_CLASSES * counts)).to(device)
  criterion = nn.MultiMarginLoss(margin=1.0, weight=w)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

  best_val_loss = float('inf')
  best_state = None

  history = {
      'train_loss' : [],
      'train_acc' : [],
      'val_loss' : [],
      'val_acc' : []
  }
  for epoch in range(1, EPOCHS+1):
    model.train()
    import random; random.shuffle(train_bags)

    epoch_loss = 0.0
    train_correct = 0

    for feat, y, _ in train_bags:
      feat = feat.to(device)
      target = torch.tensor(y, device=device)
      optimizer.zero_grad()
      scores, _ = model(feat)
      loss = criterion(scores, target)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      pred = scores.argmax(dim=1).item()
      train_correct += (pred == y)

    avg_train_loss = epoch_loss / len(train_bags)
    avg_train_acc = train_correct / len(train_bags)
    val_acc, val_loss = evaluate(model, val_bags, criterion)

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

  model.load_state_dict(best_state)
  print(f'best val loss = {best_val_loss:.4f}')
  return model, history

# 5-fold

# for fold in range(1, 6):                                                  ## OLD CODE
#   fc = f'fold_{fold}_status'                                              ## OLD CODE
#   tr_ids = meta[meta[fc] == 'train']['patient_id'].tolist()               ## OLD CODE
#   va_ids = meta[meta[fc] == 'test']['patient_id'].tolist()                ## OLD CODE

#   tr_bags = load_bags(tr_ids)                                             ## OLD CODE
#   va_bags = load_bags(va_ids)                                             ## OLD CODE

#   inst = torch.cat([b for b, _, _ in tr_bags], dim=0)                     ## OLD CODE
#   m_ = inst.mean(dim=0)                                                    ## OLD CODE
#   s_ = inst.std(dim=0) + 1e-6                                              ## OLD CODE
#   tr_bags = [((b-m_) / s_, y, pid) for b, y, pid in tr_bags]              ## OLD CODE
#   va_bags = [((b-m_) / s_, y, pid) for b, y, pid in va_bags]              ## OLD CODE

#   mdl, hist = train_model(MODE, tr_bags, va_bags, fc)                     ## OLD CODE
#   fold_models[fold] = mdl                                                 ## OLD CODE
#   fold_histories[fold] = hist                                            ## OLD CODE
#   fold_val_acc[fold] = max(hist['val_acc'])                              ## OLD CODE
#   fold_val_bags[fold] = va_bags                                          ## OLD CODE
#   print(f'[fold {fold} best val acc = {max(hist["val_acc"]):.4f}')       ## OLD CODE

custom_cv = []                                                              ## NEW CODE
for fold_name in sorted(os.listdir(CV_SPLITS_DIR)):                        ## NEW CODE
  fold_dir = os.path.join(CV_SPLITS_DIR, fold_name)                        ## NEW CODE
  with open(os.path.join(fold_dir, 'train_patients.txt')) as f:            ## NEW CODE
    train_ids = [line.strip() for line in f if line.strip()]               ## NEW CODE
  with open(os.path.join(fold_dir, 'test_patients.txt')) as f:             ## NEW CODE
    test_ids = [line.strip() for line in f if line.strip()]                ## NEW CODE
  train_ids = [pid for pid in train_ids if pid in patients]                ## NEW CODE
  test_ids = [pid for pid in test_ids if pid in patients]                  ## NEW CODE
  custom_cv.append((train_ids, test_ids))                                  ## NEW CODE

fold_histories = {}                                                         ## NEW CODE
fold_models = {}                                                            ## NEW CODE
fold_val_acc = {}                                                           ## NEW CODE
fold_val_bags = {}                                                          ## NEW CODE
fold_stats = {}                                                             ## NEW CODE

for fold, (tr_ids, va_ids) in enumerate(custom_cv, start=1):                ## NEW CODE
  tr_bags = load_bags_from_groups(tr_ids)                                  ## NEW CODE
  va_bags = load_bags_from_groups(va_ids)                                  ## NEW CODE

  inst = torch.cat([b for b, _, _ in tr_bags], dim=0)                      ## NEW CODE
  m_ = inst.mean(dim=0)                                                    ## NEW CODE
  s_ = inst.std(dim=0) + 1e-6                                              ## NEW CODE
  tr_bags = [((b-m_) / s_, y, pid) for b, y, pid in tr_bags]               ## NEW CODE
  va_bags = [((b-m_) / s_, y, pid) for b, y, pid in va_bags]               ## NEW CODE

  mdl, hist = train_model(MODE, tr_bags, va_bags)                          ## NEW CODE
  fold_models[fold] = mdl                                                  ## NEW CODE
  fold_histories[fold] = hist                                              ## NEW CODE
  fold_val_acc[fold] = max(hist['val_acc'])                                ## NEW CODE
  fold_val_bags[fold] = va_bags                                            ## NEW CODE
  fold_stats[fold] = (m_, s_)                                              ## NEW CODE
  print(f'[fold {fold} best val acc = {max(hist["val_acc"]):.4f}')         ## NEW CODE

print('------------------------------------------------------')
print(f'avg best val acc = {sum(fold_val_acc.values()) / len(fold_val_acc):.4f}')  ## NEW CODE (was hardcoded /5)



# model, history = fold_models[1], fold_histories[1]                       ## OLD CODE
model, history = fold_models[1], fold_histories[1]                         ## NEW CODE (fold 1 kept as the reference model for saving/demo below)

#평가

from sklearn.metrics import confusion_matrix, classification_report

def full_report(y_true, y_pred, name):
  print(f'==== {name} ====')
  print('혼동행렬 (행=정답, 열=예측):')
  print(confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)))
  print(classification_report(y_true, y_pred, labels=range(N_CLASSES), target_names=CLASSES, digits=3, zero_division=0))

all_true, all_pred = [], []
for fold in fold_models:                                                    ## NEW CODE (was range(1, 6); now iterate actual fold keys)
  m = fold_models[fold].eval()
  with torch.no_grad():
    for feat, y, _ in fold_val_bags[fold]:
      scores, _ = m(feat.to(device))
      all_true.append(y)
      all_pred.append(scores.argmax(dim=1).item())

full_report(all_true, all_pred, f'total 5-fold OOF ({len(all_true)})')

REPORT_PATH = os.path.join(SAVE_DIR, 'classification_report.txt')          ## NEW CODE
with open(REPORT_PATH, 'w') as f:                                            ## NEW CODE
  f.write("=== Per-fold best validation accuracy ===\n")                    ## NEW CODE
  for fold in fold_val_acc:                                                 ## NEW CODE
    f.write(f"fold {fold}: {fold_val_acc[fold]:.4f}\n")                     ## NEW CODE
  f.write(f"\naverage best val acc: {sum(fold_val_acc.values()) / len(fold_val_acc):.4f}\n\n")  ## NEW CODE

  for fold in fold_models:                                                  ## NEW CODE
    m = fold_models[fold].eval()                                            ## NEW CODE
    fold_true, fold_pred = [], []                                          ## NEW CODE
    with torch.no_grad():                                                   ## NEW CODE
      for feat, y, _ in fold_val_bags[fold]:                               ## NEW CODE
        scores, _ = m(feat.to(device))                                     ## NEW CODE
        fold_true.append(y)                                                ## NEW CODE
        fold_pred.append(scores.argmax(dim=1).item())                      ## NEW CODE
    f.write(f"=== Fold {fold} ===\n")                                      ## NEW CODE
    f.write(classification_report(fold_true, fold_pred, labels=range(N_CLASSES), target_names=CLASSES, zero_division=0))  ## NEW CODE
    f.write("\n")                                                           ## NEW CODE
  f.write("=== Aggregate (all folds combined, OOF) ===\n")                 ## NEW CODE
  f.write(classification_report(all_true, all_pred, labels=range(N_CLASSES), target_names=CLASSES, zero_division=0))  ## NEW CODE
print("saved classification report to", REPORT_PATH)                        ## NEW CODE

CONF_MATRIX_PATH = os.path.join(SAVE_DIR, 'confusion_matrix.txt')          ## NEW CODE
np.savetxt(CONF_MATRIX_PATH, confusion_matrix(all_true, all_pred, labels=range(N_CLASSES)), fmt='%d')  ## NEW CODE
print("saved confusion matrix to", CONF_MATRIX_PATH)                       ## NEW CODE

#시각화 준비

import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def collect_predictions(model, bags):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for feat, y, _ in bags:
        scores, _ = model(feat.to(device))
        prob = F.softmax(scores, dim=1)[0]
        y_true.append(y)
        y_pred.append(int(scores.argmax(dim=1)))
        y_prob.append(prob.cpu().numpy())
    import numpy as np
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

y_true_all, y_pred_all, y_prob_all = [], [], []
# for fold in range(1, 6):                                                  ## OLD CODE
for fold in fold_models:                                                    ## NEW CODE
  yt, yp, ypr = collect_predictions(fold_models[fold], fold_val_bags[fold])
  y_true_all.append(yt); y_pred_all.append(yp); y_prob_all.append(ypr)

y_true = np.concatenate(y_true_all)
y_pred = np.concatenate(y_pred_all)
y_prob = np.concatenate(y_prob_all)
print('모은 환자 수:', len(y_true))


#히트맵

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.set_yticklabels(CLASSES)
ax.set_xlabel('predicted'); ax.set_ylabel('true')

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})', ha='center', va='center',
                color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=9)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Confusion Matrix (5-fold CV)')
plt.tight_layout()
CM_PLOT_PATH = os.path.join(SAVE_DIR, 'confusion_matrix_heatmap.png')
plt.savefig(CM_PLOT_PATH, bbox_inches='tight')
print("saved confusion matrix heatmap to", CM_PLOT_PATH)
plt.show()

#시각화

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

per_class_acc = []
for c in range(N_CLASSES):
    mask = (y_true == c)
    acc = (y_pred[mask] == c).mean() if mask.sum() > 0 else 0.0
    per_class_acc.append(acc)

bars = axes[0].bar(range(N_CLASSES), per_class_acc, color='steelblue')
axes[0].set_xticks(range(N_CLASSES))
axes[0].set_xticklabels(CLASSES, rotation=45, ha='right')
axes[0].set_ylim(0, 1); axes[0].set_ylabel('accuracy')
axes[0].set_title('Per-class accuracy (Holdout)')
for b, a in zip(bars, per_class_acc):
    axes[0].text(b.get_x()+b.get_width()/2, a+0.02, f'{a:.2f}', ha='center')
axes[0].axhline(y=np.mean(per_class_acc), color='red', linestyle='--',
                label=f'mean={np.mean(per_class_acc):.2f}')
axes[0].legend()

conf = y_prob.max(axis=1)
correct = (y_true == y_pred)
axes[1].hist([conf[correct], conf[~correct]], bins=10, range=(0,1),
             label=['correct', 'wrong'], color=['green','salmon'], stacked=True)
axes[1].set_xlabel('prediction confidence (max prob)')
axes[1].set_ylabel('num patients')
axes[1].set_title('Confidence: correct vs wrong')
axes[1].legend()

plt.tight_layout()
DIAGNOSTICS_PLOT_PATH = os.path.join(SAVE_DIR, 'per_class_accuracy_and_confidence.png')
plt.savefig(DIAGNOSTICS_PLOT_PATH, bbox_inches='tight')
print("saved per-class accuracy/confidence plot to", DIAGNOSTICS_PLOT_PATH)
plt.show()

#Loss, ACC

def plot_learning_curves(history, title='', save_path=None):
  epochs = range(1, len(history['train_loss']) +1)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle(title)

  for ax, key, name in [(ax1, 'loss', 'Loss'),(ax2, 'acc', 'Accuracy')]:
    ax.plot(epochs, history[f'train_{key}'], label=f'Train {name}')
    ax.plot(epochs, history[f'val_{key}'], label=f'Val {name}')
    ax.set_title(f'Training & Validation {name}')
    ax.set_xlabel('Epochs'); ax.set_ylabel(name)
    ax.legend(); ax.grid(True)

  plt.tight_layout()
  if save_path:
    fig.savefig(save_path, bbox_inches='tight')
    print("saved learning curves plot to", save_path)
  plt.show()

plt.figure(figsize=(8, 5))
# for fold in range(1, 6):                                                  ## OLD CODE
for fold in fold_histories:                                                 ## NEW CODE
  LEARNING_CURVE_PATH = os.path.join(SAVE_DIR, f'learning_curves_fold{fold}.png')
  plot_learning_curves(fold_histories[fold], title=f'fold {fold}', save_path=LEARNING_CURVE_PATH)
  plt.plot(range(1, EPOCHS +1), fold_histories[fold]['val_acc'], label=f'fold {fold}')

  plt.title('Validation Accuracy per fold')
  plt.xlabel('Epochs'); plt.ylabel('Val Accuracy')
  plt.legend(); plt.grid(True)
  plt.tight_layout()
  VAL_ACC_OVERLAY_PATH = os.path.join(SAVE_DIR, 'val_accuracy_per_fold.png')
  plt.savefig(VAL_ACC_OVERLAY_PATH, bbox_inches='tight')
  print("saved validation accuracy per fold plot to", VAL_ACC_OVERLAY_PATH)
  plt.show()

#모델 저장

MODEL_DIR = os.path.join(SAVE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
# MODEL_PATH = os.path.join(MODEL_DIR + f'mil_svm_{MODE}_fold{FOLD}.pt')     ## OLD CODE
MODEL_PATH = os.path.join(MODEL_DIR, f'mil_svm_{MODE}.pt')                  ## NEW CODE (no single FOLD anymore, fold 1 used as reference model)

feat_mean_1, feat_std_1 = fold_stats[1]                                     ## NEW CODE (fold 1's standardization stats, matching `model` = fold_models[1])

torch.save({
    'state_dict': model.state_dict(),
    'mode': MODE,
    'in_dim': FEATURE_DIM,
    'n_classes': N_CLASSES,
    'classes': CLASSES,
    # 'feat_mean': feat_mean,                                                ## OLD CODE
    # 'feat_std': feat_std,                                                  ## OLD CODE
    'feat_mean': feat_mean_1,                                                ## NEW CODE
    'feat_std': feat_std_1,                                                  ## NEW CODE
}, MODEL_PATH)
print('저장 완료:', MODEL_PATH)

#새 데이터에 적용

def extract_one_patient_new(folder_name):                                   ## NEW CODE
  files = sorted(glob.glob(os.path.join(DATA_DIR, folder_name, '*.tif')))    ## NEW CODE
  feats, batch = [], []                                                     ## NEW CODE
  for fpath in files:                                                       ## NEW CODE
    img = Image.open(fpath).convert('RGB')                                  ## NEW CODE
    batch.append(tf(img))                                                   ## NEW CODE
    if len(batch) == BATCH_IMG:                                             ## NEW CODE
      x = torch.stack(batch).to(device)                                     ## NEW CODE
      feats.append(backbone(x).cpu())                                       ## NEW CODE
      batch = []                                                            ## NEW CODE

  if batch:                                                                 ## NEW CODE
    x = torch.stack(batch).to(device)                                      ## NEW CODE
    feats.append(backbone(x).cpu())                                        ## NEW CODE
  return torch.cat(feats, dim=0)                                            ## NEW CODE

@torch.no_grad()
def predict_patient(folder_name, model_path):
  ckpt = torch.load(model_path, map_location=device)

  net = MIL_SVM(ckpt['in_dim'], ckpt['n_classes'], mode=ckpt['mode']).to(device)
  net.load_state_dict(ckpt['state_dict'])
  net.eval()

  # feat = extract_one_patient(folder_name).to(device)                      ## OLD CODE
  feat = extract_one_patient_new(folder_name).to(device)                    ## NEW CODE
  feat = (feat - ckpt['feat_mean'].to(device)) / ckpt['feat_std'].to(device)

  scores, _ = net(feat)
  probs = torch.softmax(scores, dim=1)[0]
  pred_idx = int(scores.argmax(dim=1))
  return ckpt['classes'][pred_idx], {c: round(float(p), 3) for c, p in zip(ckpt['classes'], probs)}

#추가 데이터 분석

# pred, prob = predict_patient('cancer.CBFB_MYH11.AQK', MODEL_PATH)          ## OLD CODE
demo_folder = meta[~meta['folder'].isin(holdout_patients)]['folder'].iloc[0]  ## NEW CODE
pred, prob = predict_patient(demo_folder, MODEL_PATH)                        ## NEW CODE
print('pred class:', pred)
print('probs:', prob)
















