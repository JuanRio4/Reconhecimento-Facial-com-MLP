import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import os
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
import mediapipe as mp
from datetime import datetime
import math
import uuid
from torch.utils.tensorboard import SummaryWriter

# Configurações
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8
EXPRESSION_CLASSES = ['Feliz', 'Neutro', 'Bravo']
PATIENCE = 5
GRAD_CLIP = 1.0
K_FOLDS = 2
HAPPINESS_SPIKE_THRESHOLD = 0.6
LABEL_SMOOTHING = 0.1

# Inicializa TensorBoard
log_dir = f"runs/face_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

# Diretório para salvar os pesos
weights_save_dir = "/home/juan/Imagens/Matrizes"  # Altere para o caminho desejado
os.makedirs(weights_save_dir, exist_ok=True)  # Cria o diretório, se não existir

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {}
        label_idx = 0
        valid_extensions = ('.jpg', '.jpeg', '.png')

        print(f"Carregando dataset de: {root_dir}")
        for person in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person)
            if not os.path.isdir(person_path):
                continue
            self.label_map[label_idx] = person
            img_files = [
                f for f in os.listdir(person_path)
                if os.path.isfile(os.path.join(person_path, f))
                and f.lower().endswith(valid_extensions)
            ]

            if not img_files:
                print(f"Aviso: Nenhuma imagem válida encontrada em {person_path}")
                continue

            print(f"{person}: {len(img_files)} imagens encontradas")
            for img_name in img_files:
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Erro: Não foi possível ler a imagem {img_path}, ignorando.")
                    continue
                self.images.append(img_path)
                self.labels.append(label_idx)
            label_idx += 1

        if not self.images:
            raise ValueError("Nenhuma imagem válida foi encontrada no dataset!")

        print(f"Total de imagens carregadas: {len(self.images)}")
        print(f"Classes detectadas: {self.label_map}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Erro ao carregar imagem {img_path}")

        # Histogram equalization
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # Normalização [0, 1]
        img = (img - np.mean(img)) / np.std(img)  # Z-score

        if self.transform:
            img = self.transform(img)
            img = img.unsqueeze(0) if img.dim() == 2 else img
        else:
            img = torch.from_numpy(img).unsqueeze(0)

        return img, torch.tensor(label)

# MLP otimizada
class FaceMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaceMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Pré-processamento com alinhamento facial aprimorado
def preprocess_face(image, face_detection, face_mesh):
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return None, None

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        width, height = int(bbox.width * w), int(bbox.height * h)

        face_mesh_results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            dY = right_eye.y - left_eye.y
            dX = right_eye.x - left_eye.x
            angle = np.degrees(np.arctan2(dY, dX))

            # Centralizar e alinhar
            center = (x + width // 2, y + height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, M, (w, h))

            # Ajustar nível dos olhos
            eye_center_y = (left_eye.y + right_eye.y) / 2 * h
            target_eye_y = h * 0.4  # Olhos a 40% da altura
            translation_y = target_eye_y - eye_center_y
            M = np.float32([[1, 0, 0], [0, 1, translation_y]])
            aligned_image = cv2.warpAffine(aligned_image, M, (w, h))

            face = aligned_image[y:y+height, x:x+width]
            return face, (x, y, width, height)
    return None, None

# Reconhecimento de expressão facial
def detect_expression(image, face_mesh, prev_mouth_ratio=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            top_lip = face_landmarks.landmark[0]
            bottom_lip = face_landmarks.landmark[17]

            left_eyebrow_outer = face_landmarks.landmark[70]
            left_eyebrow_inner = face_landmarks.landmark[107]
            right_eyebrow_outer = face_landmarks.landmark[300]
            right_eyebrow_inner = face_landmarks.landmark[336]

            left_eye_upper = face_landmarks.landmark[159]
            left_eye_lower = face_landmarks.landmark[145]
            right_eye_upper = face_landmarks.landmark[386]
            right_eye_lower = face_landmarks.landmark[374]

            mouth_width = math.hypot(right_mouth.x - left_mouth.x, right_mouth.y - left_mouth.y)
            mouth_height = math.hypot(bottom_lip.y - top_lip.y, bottom_lip.x - top_lip.x)
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

            left_eyebrow_angle = np.degrees(np.arctan2(left_eyebrow_inner.y - left_eyebrow_outer.y,
                                                        left_eyebrow_inner.x - left_eyebrow_outer.x))
            right_eyebrow_angle = np.degrees(np.arctan2(right_eyebrow_inner.y - right_eyebrow_outer.y,
                                                         right_eyebrow_inner.x - right_eyebrow_outer.x))

            eyebrow_distance = math.hypot(right_eyebrow_inner.x - left_eyebrow_inner.x,
                                          right_eyebrow_inner.y - left_eyebrow_inner.y)

            left_eye_opening = math.hypot(left_eye_upper.y - left_eye_lower.y,
                                          left_eye_upper.x - left_eye_lower.x)
            right_eye_opening = math.hypot(right_eye_upper.y - right_eye_lower.y,
                                           right_eye_upper.x - right_eye_lower.x)
            avg_eye_opening = (left_eye_opening + right_eye_opening) / 2

            if prev_mouth_ratio is not None and mouth_ratio > prev_mouth_ratio + HAPPINESS_SPIKE_THRESHOLD:
                return 'Feliz'
            if mouth_ratio > 0.5 and mouth_width > 0.1:
                return 'Feliz'
            if (left_eyebrow_angle < -3 or right_eyebrow_angle < -3) and eyebrow_distance < 0.07:
                if mouth_ratio < 0.35 and avg_eye_opening < 0.03:
                    return 'Bravo'
            return 'Neutro'

    return 'Neutro'

# Transformações expandidas para data augmentation
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=10),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
])

# Função de perda com label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Treinamento com validação usando AdamW
def train_model(model, train_loader, val_loader):
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_accuracy, val_loss = evaluate_model(model, val_loader, return_loss=True)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        scheduler.step()

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Salva os pesos no diretório personalizado
            torch.save(model.state_dict(), os.path.join(weights_save_dir, 'best_face_mlp_weights.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping ativado!")
                break

    return best_accuracy

# Avaliação simplificada
def evaluate_model(model, loader, return_loss=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            if return_loss:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    if return_loss:
        return accuracy, total_loss / len(loader)
    return accuracy, all_preds, all_labels

# Validação cruzada com acurácia clara
# Validação cruzada com acurácia clara
def cross_validation(dataset):
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    accuracies = []
    all_preds = []
    all_labels = []

    # Diretório para salvar as matrizes de confusão
    matrix_save_dir = "/home/juan/Imagens/Matrizes"  # Altere para o caminho desejado
    os.makedirs(matrix_save_dir, exist_ok=True)  # Cria o diretório, se não existir

    # Configura o backend para exibição
    import matplotlib
    original_backend = matplotlib.get_backend()
    matplotlib.use('TkAgg', force=True)  # Usa TkAgg para exibição interativa

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold+1}/{K_FOLDS}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = FaceMLP(IMG_SIZE * IMG_SIZE, len(dataset.label_map)).to(DEVICE)
        accuracy = train_model(model, train_loader, val_loader)
        fold_accuracy, preds, labels = evaluate_model(model, val_loader)

        accuracies.append(fold_accuracy)
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Gera matriz de confusão para o fold
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[dataset.label_map[i] for i in range(len(dataset.label_map))],
                    yticklabels=[dataset.label_map[i] for i in range(len(dataset.label_map))])
        plt.title(f'Matriz de Confusão - Fold {fold+1}')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        # Usa backend Agg para salvar sem exibir
        matplotlib.use('Agg', force=True)
        plt.savefig(os.path.join(matrix_save_dir, f'confusion_matrix_fold_{fold+1}.png'))
        # Volta para TkAgg para exibir
        matplotlib.use('TkAgg', force=True)
        plt.show(block=False)  # Exibe sem bloquear
        plt.pause(0.1)  # Pequena pausa para garantir a renderização
        plt.close()  # Fecha a figura para evitar acúmulo

    # Restaura o backend original
    matplotlib.use(original_backend, force=True)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nAcurácia Média por Fold: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Acurácia Geral (todas as predições): {overall_accuracy:.4f}')

    # Matriz de confusão agregada
    matplotlib.use('TkAgg', force=True)  # Garante TkAgg para exibição
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[dataset.label_map[i] for i in range(len(dataset.label_map))],
                yticklabels=[dataset.label_map[i] for i in range(len(dataset.label_map))])
    plt.title('Matriz de Confusão Agregada')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    # Usa backend Agg para salvar
    matplotlib.use('Agg', force=True)
    plt.savefig(os.path.join(matrix_save_dir, 'confusion_matrix_aggregated.png'))
    # Volta para TkAgg para exibir
    matplotlib.use('TkAgg', force=True)
    plt.show(block=False)  # Exibe sem bloquear
    plt.pause(0.1)  # Pequena pausa para renderização
    plt.close()  # Fecha a figura

    # Restaura o backend original
    matplotlib.use(original_backend, force=True)

    return overall_accuracy
# Predição de uma imagem
def predict_face(image, model, face_detection, face_mesh):
    face, bbox = preprocess_face(image, face_detection, face_mesh)
    if face is None:
        return "No face detected", None, 0.0

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face_gray = face_gray.astype(np.float32) / 255.0
    face_gray = (face_gray - np.mean(face_gray)) / np.std(face_gray)
    face_tensor = torch.from_numpy(face_gray).unsqueeze(0).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(face_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        confidence = confidence.item()

        if confidence < CONFIDENCE_THRESHOLD:
            return "Desconhecido", bbox, confidence
        label = dataset.label_map[pred.item()]

    return label, bbox, confidence

# Processamento de imagem de arquivo
def process_image_file(file_path, model, face_detection, face_mesh):
    if not os.path.exists(file_path):
        print("Arquivo não encontrado!")
        return

    image = cv2.imread(file_path)
    if image is None:
        print("Erro ao carregar a imagem!")
        return

    label, bbox, confidence = predict_face(image, model, face_detection, face_mesh)
    expression = detect_expression(image, face_mesh)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f}) - {expression}"
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_path = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, image)
    print(f"Resultado salvo em: {output_path}")

    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Reconhecimento em tempo real
def real_time_recognition(model, face_detection, face_mesh):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return

    prev_mouth_ratio = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, bbox, confidence = predict_face(frame, model, face_detection, face_mesh)
        expression = detect_expression(frame, face_mesh, prev_mouth_ratio)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            left_mouth = results.multi_face_landmarks[0].landmark[61]
            right_mouth = results.multi_face_landmarks[0].landmark[291]
            top_lip = results.multi_face_landmarks[0].landmark[0]
            bottom_lip = results.multi_face_landmarks[0].landmark[17]
            mouth_width = math.hypot(right_mouth.x - left_mouth.x, right_mouth.y - left_mouth.y)
            mouth_height = math.hypot(bottom_lip.y - top_lip.y, bottom_lip.x - top_lip.x)
            prev_mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({confidence:.2f}) - {expression}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Reconhecimento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Função principal
def main():
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

    global dataset
    dataset = FaceDataset('/home/juan/Área de Trabalho/dataset_treino', transform=data_transforms)
    test_dataset = FaceDataset('/home/juan/Área de Trabalho/dataset_teste')
    dataset.images.extend(test_dataset.images)
    dataset.labels.extend(test_dataset.labels)
    dataset.label_map = {**dataset.label_map, **{k+len(dataset.label_map): v for k, v in test_dataset.label_map.items()}}

    print("\n=== Resumo do Dataset Combinado ===")
    print(f"Total de imagens: {len(dataset)}")
    print(f"Classes: {dataset.label_map}")

    overall_accuracy = cross_validation(dataset)

    model = FaceMLP(IMG_SIZE * IMG_SIZE, len(dataset.label_map)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(weights_save_dir, 'best_face_mlp_weights.pth')))
    model.eval()

    # Salva os pesos finais no diretório personalizado
    final_weights_path = os.path.join(weights_save_dir, 'face_mlp_final_weights.pth')
    torch.save(model.state_dict(), final_weights_path)
    print(f"\nPesos finais da rede salvos em: {final_weights_path}")

    while True:
        print("\n=== Sistema de Reconhecimento Facial ===")
        print("1. Reconhecer rosto por arquivo de imagem")
        print("2. Reconhecer rosto em tempo real (webcam)")
        print("3. Sair")
        choice = input("Escolha uma opção (1-3): ")

        if choice == '1':
            file_path = input("Digite o caminho do arquivo de imagem: ")
            process_image_file(file_path, model, face_detection, face_mesh)
        elif choice == '2':
            real_time_recognition(model, face_detection, face_mesh)
        elif choice == '3':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")

    face_detection.close()
    face_mesh.close()
    writer.close()

if __name__ == "__main__":
    main()