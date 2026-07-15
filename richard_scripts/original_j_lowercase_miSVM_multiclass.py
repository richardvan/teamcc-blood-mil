#구글드라이브 연결, 압축해제

from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

SAVE_DIR = '/content/drive/MyDrive/SVM'
DATA_DIR = '/content'
ZIP_PATH = '/content/drive/MyDrive/data.zip'

expected_dir = os.path.join(DATA_DIR, 'CBFB_MYH11')

if not os.path.exists(expected_dir):
  with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall('/content')
  print('압축 해제 완료')

else:
  print('이미 압축 해제됨')

assert os.path.isdir('/content/CBFB_MYH11/AQK'), '폴더가 없습니다: /content/CBFB_MYH11/AQK'

print(len([
    f for f in os.listdir('/content/CBFB_MYH11/AQK')
    if f.lower().endswith('.tif')
]))

#설정값

import torch

CLASSES = ['control', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

FOLD = 1
MODE = 'mi'

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

FEAT_CACHE = SAVE_DIR + '/features/'
os.makedirs(FEAT_CACHE, exist_ok=True)

def extract_one_patient(folder_name):
  parts = folder_name.split('.')
  folder = os.path.join(DATA_DIR, parts[1], parts[2])
  files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.tif'))
  feats, batch = [], []
  for f in files:
    img = Image.open(os.path.join(folder, f)).convert('RGB')
    batch.append(tf(img))
    if len(batch) == BATCH_IMG:
      x = torch.stack(batch).to(device)
      feats.append(backbone(x).cpu())
      batch = []

  if batch:
   x = torch.stack(batch).to(device)
   feats.append(backbone(x).cpu())
  return torch.cat(feats, dim=0)

#메타데이터 추출

import pandas as pd

meta = pd.read_csv(SAVE_DIR + '/metadata_server.csv')

for _, row in meta.iterrows():
  pid = row['patient_id']
  fpath = FEAT_CACHE + f'{pid}.pt'
  if os.path.exists(fpath):
    continue
  feat = extract_one_patient(row['folder'])
  torch.save(feat, fpath)
  print(f'{pid}: {tuple(feat.shape)} 저장')

print('특징 추출/캐시 완료')

#라벨, 폴드

def folder_to_label(folder):
  subtype = folder.split('.')[1]
  return CLASS_TO_IDX[subtype]

meta['label'] = meta['folder'].apply(folder_to_label)

fold_col = f'fold_{FOLD}_status'

train_ids = meta[meta[fold_col] == 'train']['patient_id'].tolist()
val_ids = meta[meta[fold_col] == 'test']['patient_id'].tolist()
holdout_ids = meta[meta['is_holdout'] == True]['patient_id'].tolist()

print(f'train {len(train_ids)}명 / val {len(val_ids)}명 / holdout {len(holdout_ids)}명')
for k in range(1, 6):
  fc = f'fold_{k}_status'
  n_train = (meta[fc] == 'train').sum()
  n_val = (meta[fc] == 'test').sum()
  dist = meta[meta[fc] == 'train']['label'].value_counts().sort_index().tolist()
  print(f'[fold {k}] train {n_train} / val {n_val} | {dist}')

#bag 데이터 준비, 표준화

def load_bags(id_list):
  label_map = dict(zip(meta['patient_id'], meta['label']))
  bags = []
  for pid in id_list:
    feat = torch.load(FEAT_CACHE + f'{pid}.pt')
    bags.append((feat, label_map[pid], pid))
  return bags

train_bags = load_bags(train_ids)
val_bags = load_bags(val_ids)
holdout_bags = load_bags(holdout_ids)

train_instances = torch.cat([f for f, _, _ in train_bags], dim=0)
feat_mean = train_instances.mean(dim=0)
feat_std = train_instances.std(dim=0) + 1e-6

def standardize(bags):
  return [((f-feat_mean) / feat_std, y, pid) for f, y, pid in bags]

train_bags = standardize(train_bags)
val_bags = standardize(val_bags)
holdout_bags = standardize(holdout_bags)

print('표준화 완료')

#MIL_SVM

torch.manual_seed(SEED)

class MIL_SVM(nn.Module):
  def __init__(self, in_dim, n_classes, mode='mi'):
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

def train_model(mode, train_bags, val_bags, fold_col):
  model = MIL_SVM(FEATURE_DIM, N_CLASSES, mode=mode).to(device)
  counts = torch.tensor([ (meta[meta[fold_col]=='train']['label']==i).sum() for i in range(N_CLASSES)], dtype=torch.float)
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

fold_histories = {}
fold_models = {}
fold_val_acc = {}
fold_val_bags = {}

for fold in range(1, 6):
  fc = f'fold_{fold}_status'
  tr_ids = meta[meta[fc] == 'train']['patient_id'].tolist()
  va_ids = meta[meta[fc] == 'test']['patient_id'].tolist()

  tr_bags = load_bags(tr_ids)
  va_bags = load_bags(va_ids)

  inst = torch.cat([b for b, _, _ in tr_bags], dim=0)
  m_ = inst.mean(dim=0)
  s_ = inst.std(dim=0) + 1e-6
  tr_bags = [((b-m_) / s_, y, pid) for b, y, pid in tr_bags]
  va_bags = [((b-m_) / s_, y, pid) for b, y, pid in va_bags]

  mdl, hist = train_model(MODE, tr_bags, va_bags, fc)
  fold_models[fold] = mdl
  fold_histories[fold] = hist
  fold_val_acc[fold] = max(hist['val_acc'])
  fold_val_bags[fold] = va_bags
  print(f'[fold {fold} best val acc = {max(hist["val_acc"]):.4f}')

print('------------------------------------------------------')
print(f'avg best val acc = {sum(fold_val_acc.values()) / 5:.4f}')



model, history = fold_models[1], fold_histories[1]

#평가

from sklearn.metrics import confusion_matrix, classification_report

def full_report(y_true, y_pred, name):
  print(f'==== {name} ====')
  print('혼동행렬 (행=정답, 열=예측):')
  print(confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)))
  print(classification_report(y_true, y_pred, labels=range(N_CLASSES), target_names=CLASSES, digits=3, zero_division=0))

all_true, all_pred = [], []
for fold in range(1, 6):
  m = fold_models[fold].eval()
  with torch.no_grad():
    for feat, y, _ in fold_val_bags[fold]:
      scores, _ = m(feat.to(device))
      all_true.append(y)
      all_pred.append(scores.argmax(dim=1).item())

full_report(all_true, all_pred, f'total 5-fold OOF ({len(all_true)})')

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
for fold in range(1, 6):
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

plt.tight_layout(); plt.show()

#Loss, ACC

def plot_learning_curves(history, title=''):
  epochs = range(1, len(history['train_loss']) +1)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle(title)

  for ax, key, name in [(ax1, 'loss', 'Loss'),(ax2, 'acc', 'Accuracy')]:
    ax.plot(epochs, history[f'train_{key}'], label=f'Train {name}')
    ax.plot(epochs, history[f'val_{key}'], label=f'Val {name}')
    ax.set_title(f'Training & Validation {name}')
    ax.set_xlabel('Epochs'); ax.set_ylabel(name)
    ax.legend(); ax.grid(True)

  plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
for fold in range(1, 6):
  plot_learning_curves(fold_histories[fold], title=f'fold {fold}')
  plt.plot(range(1, EPOCHS +1), fold_histories[fold]['val_acc'], label=f'fold {fold}')

  plt.title('Validation Accuracy per fold')
  plt.xlabel('Epochs'); plt.ylabel('Val Accuracy')
  plt.legend(); plt.grid(True)
  plt.tight_layout(); plt.show()

#모델 저장

MODEL_DIR = os.path.join(SAVE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR + f'mil_svm_{MODE}_fold{FOLD}.pt')

torch.save({
    'state_dict': model.state_dict(),
    'mode': MODE,
    'in_dim': FEATURE_DIM,
    'n_classes': N_CLASSES,
    'classes': CLASSES,
    'feat_mean': feat_mean,
    'feat_std': feat_std,
}, MODEL_PATH)
print('저장 완료:', MODEL_PATH)

#새 데이터에 적용

@torch.no_grad()
def predict_patient(folder_name, model_path):
  ckpt = torch.load(model_path, map_location=device)

  net = MIL_SVM(ckpt['in_dim'], ckpt['n_classes'], mode=ckpt['mode']).to(device)
  net.load_state_dict(ckpt['state_dict'])
  net.eval()

  feat = extract_one_patient(folder_name).to(device)
  feat = (feat - ckpt['feat_mean'].to(device)) / ckpt['feat_std'].to(device)

  scores, _ = net(feat)
  probs = torch.softmax(scores, dim=1)[0]
  pred_idx = int(scores.argmax(dim=1))
  return ckpt['classes'][pred_idx], {c: round(float(p), 3) for c, p in zip(ckpt['classes'], probs)}

#추가 데이터 분석

pred, prob = predict_patient('cancer.CBFB_MYH11.AQK', MODEL_PATH)
print('pred class:', pred)
print('probs:', prob)
















