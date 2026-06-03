import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def nms(boxes, scores, threshold=0.5):
    """
    非极大值抑制
    boxes: [[cx, cy, w, h], ...] 归一化坐标
    scores: [0.9, 0.8, ...]
    """
    if len(boxes) == 0:
        return []

    # 转成 [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算当前框与其余框的IOU
        current_box = [x1[current], y1[current], x2[current], y2[current]]
        other_indices = indices[1:]

        xx1 = np.maximum(current_box[0], x1[other_indices])
        yy1 = np.maximum(current_box[1], y1[other_indices])
        xx2 = np.minimum(current_box[2], x2[other_indices])
        yy2 = np.minimum(current_box[3], y2[other_indices])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[current] + areas[other_indices] - inter_area
        ious = inter_area / union_area

        # 保留IOU小的
        mask = ious < threshold
        indices = other_indices[mask]

    return keep
# ==================== 可视化工具类 ====================
class SimpleTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.ious = []
        self.cls_acc = []

    def add(self, train_loss, val_loss, iou, cls_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.ious.append(iou)
        self.cls_acc.append(cls_acc)

    def draw(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.train_losses, label='训练Loss', color='blue')
        axes[0].plot(self.val_losses, label='验证Loss', color='red')
        axes[0].set_title('Loss变化')
        axes[0].set_xlabel('轮数')
        axes[0].set_ylabel('Loss值')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.ious, label='IoU', color='green')
        axes[1].set_title('定位准确度')
        axes[1].set_xlabel('轮数')
        axes[1].set_ylabel('IoU值(0-1)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.cls_acc, label='缺陷检测准确率', color='orange')
        axes[2].set_title('缺陷分类准确率')
        axes[2].set_xlabel('轮数')
        axes[2].set_ylabel('准确率')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('训练结果.png', dpi=150)
        plt.show()
        print("✅ 图片已保存为：训练结果.png")


# ---------------------- 1. 全局配置 ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMG_DIR = r"/2/defect_dataset/train/images"
TRAIN_LABEL_DIR = r"/2/defect_dataset/train/labels"
VAL_IMG_DIR = r"/2/defect_dataset/val/images"
VAL_LABEL_DIR = r"/2/defect_dataset/val/labels"
TEST_IMG_DIR = r"/2/defect_dataset/test/images"
TEST_LABEL_DIR = r"/2/defect_dataset/test/labels"

MODEL_SAVE_PATH = r"../model/defect_detection_bbox.pth"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001


# ---------------------- 2. 模型定义 ----------------------
class DefectBboxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        # 去掉 avgpool 和 fc，保留空间特征
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 输出: [B, 512, 7, 7] 对于 224x224 输入

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 分类头自己加池化
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 回归头自己加池化
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.cnn_backbone(x)  # [B, 512, 7, 7]
        cls_pred = self.classifier(features)
        bbox_pred = self.bbox_regressor(features)
        return cls_pred, bbox_pred

    def forward_features(self, x):
        """新增：返回空间特征给 RPN 用"""
        return self.cnn_backbone(x)  # [B, 512, 7, 7]IOU不足0.3，效果不理想



# ---------------------- 3. 数据集类 ----------------------
class DefectBboxDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mode = mode

        self.crop_size = 256
        self.input_size = 224

        self.img_paths = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.img_paths.append(os.path.join(root, file))

        self.data = []
        for img_path in self.img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(label_dir, f"{img_name}.txt")

            defect_exist = 0
            bbox = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                defect_exist = 1
                with open(txt_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip().split()
                    cx, cy, w, h = map(float, line[1:5])
                    bbox = np.array([cx, cy, w, h], dtype=np.float32)

            self.data.append((img_path, defect_exist, bbox))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, defect_exist, bbox = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        orig_w, orig_h = img.size

        if defect_exist == 1:
            cx_norm, cy_norm = bbox[0], bbox[1]
            cx_pixel = int(cx_norm * orig_w)
            cy_pixel = int(cy_norm * orig_h)

            left = max(0, min(cx_pixel - self.crop_size // 2, orig_w - self.crop_size))
            top = max(0, min(cy_pixel - self.crop_size // 2, orig_h - self.crop_size))
            right = left + self.crop_size
            bottom = top + self.crop_size

            img_cropped = img.crop((left, top, right, bottom))

            new_cx = (cx_pixel - left) / self.crop_size
            new_cy = (cy_pixel - top) / self.crop_size
            new_w = bbox[2] * orig_w / self.crop_size
            new_h = bbox[3] * orig_h / self.crop_size

            new_w = min(new_w, 1.0)
            new_h = min(new_h, 1.0)

            bbox = np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)

        else:
            if self.mode == 'train':
                left = np.random.randint(0, max(1, orig_w - self.crop_size))
                top = np.random.randint(0, max(1, orig_h - self.crop_size))
            else:
                left = (orig_w - self.crop_size) // 2
                top = (orig_h - self.crop_size) // 2

            img_cropped = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        img_resized = img_cropped.resize((self.input_size, self.input_size))

        if self.transform:
            img_resized = self.transform(img_resized)

        return img_resized, torch.tensor(defect_exist, dtype=torch.float32), torch.tensor(bbox, dtype=torch.float32)


# ---------------------- 4. IoU计算 ----------------------
def calculate_iou(bbox1, bbox2, img_size):
    def norm2pixel(bbox, size):
        cx, cy, w, h = bbox
        xmin = (cx - w / 2) * size
        ymin = (cy - h / 2) * size
        xmax = (cx + w / 2) * size
        ymax = (cy + h / 2) * size
        return np.clip([xmin, ymin, xmax, ymax], 0, size)

    x1, y1, x2, y2 = norm2pixel(bbox1, img_size)
    x1t, y1t, x2t, y2t = norm2pixel(bbox2, img_size)

    inter_x1 = max(x1, x1t)
    inter_y1 = max(y1, y1t)
    inter_x2 = min(x2, x2t)
    inter_y2 = min(y2, y2t)
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    pred_area = (x2 - x1) * (y2 - y1)
    true_area = (x2t - x1t) * (y2t - y1t)
    union_area = pred_area + true_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ---------------------- 5. 训练函数 ----------------------
def train_bbox_model(train_loader, val_loader, model, epochs=EPOCHS):
    cls_criterion = nn.BCELoss()
    bbox_criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_iou = 0.0

    tracker = SimpleTracker()
    pbar = tqdm(range(epochs), desc='Training')

    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for imgs, defect_exist, bbox_labels in train_loader:
            imgs = imgs.to(DEVICE)
            defect_exist = defect_exist.to(DEVICE).unsqueeze(1)
            bbox_labels = bbox_labels.to(DEVICE)

            cls_pred, bbox_pred = model(imgs)

            cls_loss = cls_criterion(cls_pred, defect_exist)
            bbox_loss = bbox_criterion(bbox_pred, bbox_labels)
            loss = cls_loss + bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        cls_correct = 0
        total_iou = 0.0
        defect_count = 0
        total_samples = 0

        with torch.no_grad():
            for imgs, defect_exist, bbox_labels in val_loader:
                imgs = imgs.to(DEVICE)
                defect_exist = defect_exist.to(DEVICE).unsqueeze(1)
                bbox_labels = bbox_labels.to(DEVICE)

                cls_pred, bbox_pred = model(imgs)
                bbox_pred = torch.clamp(bbox_pred, 0, 1)
                cls_loss = cls_criterion(cls_pred, defect_exist)
                bbox_loss = bbox_criterion(bbox_pred, bbox_labels)
                val_loss += (cls_loss + bbox_loss).item()

                cls_pred_binary = (cls_pred > 0.5).float()
                cls_correct += (cls_pred_binary == defect_exist).sum().item()

                for i in range(len(defect_exist)):
                    if defect_exist[i] == 1:
                        pred_bbox = bbox_pred[i].cpu().numpy()
                        true_bbox = bbox_labels[i].cpu().numpy()
                        total_iou += calculate_iou(pred_bbox, true_bbox, IMG_SIZE)
                        defect_count += 1
                total_samples += len(defect_exist)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        cls_acc = cls_correct / total_samples * 100
        avg_iou = total_iou / defect_count if defect_count > 0 else 0

        tracker.add(avg_train_loss, avg_val_loss, avg_iou, cls_acc)

        pbar.set_postfix({
            'loss': f'{avg_val_loss:.4f}',
            'cls_acc': f'{cls_acc:.1f}%',
            'IoU': f'{avg_iou:.4f}'
        })

        if avg_iou > best_iou:
            best_iou = avg_iou
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    tracker.draw()
    print(f"最佳验证集 IoU: {best_iou:.4f}")
    return model


# ---------------------- 6. 可视化预测结果 ----------------------
def visualize_predictions(model, dataset, num_samples=5, save_path='predictions.png'):
    model.eval()
    defect_indices = [i for i in range(len(dataset)) if dataset.data[i][1] == 1]
    if len(defect_indices) == 0:
        print("警告：数据集中没有缺陷样本！")
        return

    selected_indices = np.random.choice(defect_indices, min(num_samples, len(defect_indices)), replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, ax in zip(selected_indices, axes):
            img, defect_exist, true_bbox = dataset[idx]

            img_batch = img.unsqueeze(0).to(DEVICE)
            _, bbox_pred = model(img_batch)
            pred_bbox = bbox_pred[0].cpu().numpy()

            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)

            ax.imshow(img_display)
            ax.set_title(f'样本 {idx}')
            ax.axis('off')

            cx, cy, w, h = true_bbox.numpy()
            x1 = (cx - w / 2) * IMG_SIZE
            y1 = (cy - h / 2) * IMG_SIZE
            rect_true = patches.Rectangle((x1, y1), w * IMG_SIZE, h * IMG_SIZE,
                                          linewidth=2, edgecolor='red', facecolor='none', label='真实框')
            ax.add_patch(rect_true)

            cx_p, cy_p, w_p, h_p = pred_bbox
            x1_p = (cx_p - w_p / 2) * IMG_SIZE
            y1_p = (cy_p - h_p / 2) * IMG_SIZE
            rect_pred = patches.Rectangle((x1_p, y1_p), w_p * IMG_SIZE, h_p * IMG_SIZE,
                                          linewidth=2, edgecolor='green', facecolor='none', label='预测框',
                                          linestyle='--')
            ax.add_patch(rect_pred)

            sample_iou = calculate_iou(pred_bbox, true_bbox.numpy(), IMG_SIZE)
            ax.text(5, 15, f'IoU: {sample_iou:.3f}', color='yellow', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label='真实框'),
                       Line2D([0], [0], color='green', lw=2, linestyle='--', label='预测框')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 可视化结果已保存：{save_path}")


# ---------------------- 7. 测试函数 ----------------------
def evaluate_model(test_loader, model):
    model.eval()
    cls_correct = 0
    total_iou = 0.0
    defect_count = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, defect_exist, bbox_labels in test_loader:
            imgs = imgs.to(DEVICE)
            defect_exist = defect_exist.to(DEVICE).unsqueeze(1)
            bbox_labels = bbox_labels.to(DEVICE)

            cls_pred, bbox_pred = model(imgs)

            cls_pred_binary = (cls_pred > 0.5).float()
            cls_correct += (cls_pred_binary == defect_exist).sum().item()

            for i in range(len(defect_exist)):
                if defect_exist[i] == 1:
                    pred_bbox = bbox_pred[i].cpu().numpy()
                    true_bbox = bbox_labels[i].cpu().numpy()
                    total_iou += calculate_iou(pred_bbox, true_bbox, IMG_SIZE)
                    defect_count += 1
            total_samples += len(defect_exist)

    cls_acc = cls_correct / total_samples * 100
    avg_iou = total_iou / defect_count if defect_count > 0 else 0
    print(f"测试集缺陷检测准确率：{cls_acc:.2f}%")
    print(f"测试集定位IoU：{avg_iou:.4f}")


# ---------------------- 8. 主程序 ----------------------
if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("正在加载数据集...")
    train_dataset = DefectBboxDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, train_transform, mode='train')
    val_dataset = DefectBboxDataset(VAL_IMG_DIR, VAL_LABEL_DIR, val_test_transform, mode='val')
    test_dataset = DefectBboxDataset(TEST_IMG_DIR, TEST_LABEL_DIR, val_test_transform, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DefectBboxCNN().to(DEVICE)

    print("加载预训练权重...")
    pretrained_model = resnet18(weights="IMAGENET1K_V1")
    state_dict = pretrained_model.state_dict()
    del state_dict["fc.weight"], state_dict["fc.bias"]
    model.cnn_backbone.load_state_dict(state_dict, strict=False)

    for name, param in model.cnn_backbone.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(f"\n训练集: {len(train_dataset)}张, 验证集: {len(val_dataset)}张, 测试集: {len(test_dataset)}张")

    print("开始训练...")
    model = train_bbox_model(train_loader, val_loader, model)

    print(f"\n加载最佳模型：{MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_model(test_loader, model)

    print("\n生成预测可视化图...")
    visualize_predictions(model, test_dataset, num_samples=3)