import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# БЛОК 1: ПІДГОТОВКА ДАНИХ
# =====================================================================
class DataManager:
    @staticmethod
    def get_image_data(data_dir, batch_size=32):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        print("Знайдено класи:", dataset.classes)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, dataset.classes

    @staticmethod
    def get_tabular_data(csv_path, target_col):
        df = pd.read_csv(csv_path)

        X = df.drop(columns=[target_col]).values
        Y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        unique_classes = np.unique(Y)
        classes = [f"Клас {c}" for c in unique_classes]

        return train_loader, test_loader, classes

    @staticmethod
    def get_timeseries_data(csv_path, target_col, seq_length=20):
        df = pd.read_csv(csv_path)
        data = df[target_col].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(data.reshape(-1, 1))

        def create_sequences(data, seq_length):
            xs = []
            ys = []
            for i in range(len(data) - seq_length):
                x = data[i:(i + seq_length)]
                y = data[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        X, y = create_sequences(data_normalized, seq_length)

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, X_test, y_train, y_test, scaler

# =====================================================================
# БЛОК 2: АРХІТЕКТУРИ МОДЕЛЕЙ
# =====================================================================

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout_prob=0.3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout_prob=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size,  16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear( 8, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x



# =====================================================================
# БЛОК 3: НАВЧАННЯ
# =====================================================================
class Trainer:
    @staticmethod
    def train_classifier(model, train_loader, test_loader, epochs=30, lr=0.001):
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        all_preds = []
        all_labels = []

        print(f"\n[КЛАСИФІКАЦІЯ] Початок навчання на {epochs} епох...")
        for epoch in range(epochs):
            # --- ТРЕНУВАННЯ ---
            model.train()
            running_train_loss = 0.0
            correct_train, total_train = 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_train_acc = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # --- ТЕСТУВАННЯ ---
            model.eval()
            running_test_loss = 0.0
            correct_test, total_test = 0, 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

                    # Збираємо передбачення для матриці помилок на останній епосі
                    if epoch == epochs - 1:
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

            epoch_test_loss = running_test_loss / len(test_loader)
            epoch_test_acc = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_acc)

            # Виводимо прогрес
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Епоха [{epoch + 1}/{epochs}] | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Test Loss: {epoch_test_loss:.4f}, Acc: {epoch_test_acc:.4f}")

        return train_losses, test_losses, train_accuracies, test_accuracies, all_labels, all_preds

    @staticmethod
    def train_regressor(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.001, patience=20):
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 Регуляризація з Лаб 8

        train_losses, test_losses = [], []

        # Для Ранньої Зупинки
        best_test_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        print(f"\n[ПРОГНОЗУВАННЯ] Початок навчання на {epochs} епох...")
        for epoch in range(epochs):
            # --- ТРЕНУВАННЯ ---
            model.train()
            inputs, targets = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # --- ТЕСТУВАННЯ ---
            model.eval()
            with torch.no_grad():
                test_inputs, test_targets = X_test.to(device), y_test.to(device)
                test_outputs = model(test_inputs)
                t_loss = criterion(test_outputs, test_targets)
                test_losses.append(t_loss.item())

            # Рання зупинка
            if t_loss.item() < best_test_loss:
                best_test_loss = t_loss.item()
                epochs_no_improve = 0
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f'Епоха [{epoch + 1}/{epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {t_loss.item():.4f}')

            if epochs_no_improve >= patience:
                print(f'-> Рання зупинка на епосі {epoch + 1}! Тестова помилка не покращувалась {patience} епох.')
                break

        # Відновлюємо найкращі ваги
        model.load_state_dict(best_model_wts)
        return train_losses, test_losses


# =====================================================================
# БЛОК 4: ВІЗУАЛІЗАЦІЯ
# =====================================================================
class Visualizer:
    @staticmethod
    def plot_learning_curves(train_losses, test_losses, train_acc=None, test_acc=None):
        if train_acc and test_acc:
            # Для задач класифікації (Малюємо Loss і Accuracy)
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
            plt.plot(test_losses, label='Test Loss', color='red', marker='s', markersize=3)
            plt.title('Графік функції втрат (Loss)')
            plt.xlabel('Епохи')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label='Train Accuracy', color='blue', marker='o', markersize=3)
            plt.plot(test_acc, label='Test Accuracy', color='red', marker='s', markersize=3)
            plt.title('Графік точності (Accuracy)')
            plt.xlabel('Епохи')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        else:
            # Для задач часових рядів (Малюємо тільки Loss)
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label='Train Loss', color='blue')
            plt.plot(test_losses, label='Test Loss', color='red')
            plt.title('Крива навчання (MSE Loss)')
            plt.xlabel('Епохи')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
        plt.title('Матриця помилок (Confusion Matrix)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_classification_metrics(y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [precision, recall, f1]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics, values, color=['#4C72B0', '#55A868', '#C44E52'])
        plt.ylim(0, 1.1)
        plt.title('Оцінка якості класифікації (Метрики)')
        plt.ylabel('Значення (від 0 до 1)')

        # Додаємо цифри над стовпчиками
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom',
                     fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_time_series_forecast(y_real, y_pred_rnn, y_pred_lstm):
        plt.figure(figsize=(12, 5))
        display_points = 150  # Показуємо останні 150 точок

        plt.plot(y_real[-display_points:], label='Справжні дані (Real)', color='green', linewidth=2)

        if y_pred_rnn is not None:
            plt.plot(y_pred_rnn[-display_points:], label='Прогноз RNN', color='blue', linestyle='--')
        if y_pred_lstm is not None:
            plt.plot(y_pred_lstm[-display_points:], label='Прогноз LSTM', color='red', linestyle='--')

        plt.title('Прогнозування часового ряду на тестових даних')
        plt.xlabel('Часовий крок')
        plt.ylabel('Значення')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =====================================================================
# БЛОК 5: ІНТЕРФЕЙС
# =====================================================================
if __name__ == "__main__":
    print("===========================================")
    print("VERY SUPER BEAUTIFUL POWERFUL MEGA SUCCESSFUL GRATEFUL CORRECT GREATLY PROGRAM FOR Artificial intelligence IN THE WORLD!!!")
    print("Thank you for your attention for this matter.")
    print("===========================================")
    print("Оберіть тип завдання, який хочете запустити:")
    print("  [1] Класифікація зображень (CNN)")
    print("  [2] Класифікація табличних даних (MLP)")
    print("  [3] Прогнозування часових рядів (RNN & LSTM)")
    print("  [0] Вихід")
    print("===========================================")
    data_path = None
    target_column = None
    choice = input("Ваш вибір (0-3): ").strip()


    if choice in ["1", "2", "3"]:
        data_path = input("Введіть шлях до датасету (назва папки або файлу .csv): ").strip()

        if choice in ["2", "3"]:
            target_column = input("Введіть назву цільової колонки (target column) у вашому CSV: ").strip()

    match choice:
        case "1":
            print(f"\n--- ЗАПУСК: Класифікація зображень (Папка: {data_path}) ---")
            try:
                train_loader, test_loader, classes = DataManager.get_image_data(data_path, batch_size=32)
                model_cnn = CNN(num_classes=len(classes))

                tr_loss, te_loss, tr_acc, te_acc, labels, preds = Trainer.train_classifier(
                    model_cnn, train_loader, test_loader, epochs=30, lr=0.001
                )

                Visualizer.plot_learning_curves(tr_loss, te_loss, tr_acc, te_acc)
                Visualizer.plot_classification_metrics(labels, preds)
                Visualizer.plot_confusion_matrix(labels, preds, classes)
            except FileNotFoundError:
                print(f"Помилка: Папку '{data_path}' не знайдено. Перевірте правильність назви.")

        case "2":
            print(f"\n--- ЗАПУСК: Класифікація табличних даних (Файл: {data_path}) ---")
            try:
                train_loader, test_loader, classes = DataManager.get_tabular_data(data_path, target_col=target_column)

                sample_x, _ = next(iter(train_loader))
                dynamic_input_size = sample_x.shape[1]
                print(f"Автоматично визначено вхідних ознак: {dynamic_input_size}")

                model_mlp = MLP(input_size=dynamic_input_size, num_classes=len(classes))

                tr_loss, te_loss, tr_acc, te_acc, labels, preds = Trainer.train_classifier(
                    model_mlp, train_loader, test_loader, epochs=200, lr=0.001
                )

                Visualizer.plot_learning_curves(tr_loss, te_loss, tr_acc, te_acc)
                Visualizer.plot_classification_metrics(labels, preds)
                Visualizer.plot_confusion_matrix(labels, preds, classes)
            except FileNotFoundError:
                print(f"Помилка: Файл '{data_path}' не знайдено.")
            except KeyError:
                print(f"Помилка: Колонку '{target_column}' не знайдено у файлі. Перевірте правильність назви.")

        case "3":
            print(f"\n--- ЗАПУСК: Прогнозування часових рядів (Файл: {data_path}) ---")
            try:
                X_train, X_test, y_train, y_test, scaler = DataManager.get_timeseries_data(data_path, target_col=target_column, seq_length=20)
                print("\n[Тренування RNN]")
                model_rnn = RNN()
                rnn_tr_loss, rnn_te_loss = Trainer.train_regressor(model_rnn, X_train, y_train, X_test, y_test, epochs=150)

                model_rnn.eval()
                with torch.no_grad():
                    rnn_preds = scaler.inverse_transform(model_rnn(X_test.to(device)).cpu().numpy())

                print("\n[Тренування LSTM]")
                model_lstm = LSTM()
                lstm_tr_loss, lstm_te_loss = Trainer.train_regressor(model_lstm, X_train, y_train, X_test, y_test, epochs=150)

                model_lstm.eval()
                with torch.no_grad():
                    lstm_preds = scaler.inverse_transform(model_lstm(X_test.to(device)).cpu().numpy())

                y_real = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

                Visualizer.plot_learning_curves(rnn_tr_loss, rnn_te_loss)
                Visualizer.plot_learning_curves(lstm_tr_loss, lstm_te_loss)
                Visualizer.plot_time_series_forecast(y_real, rnn_preds, lstm_preds)
            except FileNotFoundError:
                print(f"Помилка: Файл '{data_path}' не знайдено.")
            except KeyError:
                print(f"Помилка: Колонку '{target_column}' не знайдено у файлі. Перевірте правильність назви.")

        case "0":
            print("Вихід з програми. До побачення!")

        case _:
            print("Невірний вибір. Будь ласка, запустіть програму знову та оберіть цифру від 0 до 3.")