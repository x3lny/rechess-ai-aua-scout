import torch
import torch.nn as nn
import torch.optim as optim


class ChessResBlock(nn.Module):
    def __init__(self, channels):
        super(ChessResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# ВСТАВЬ ЭТОТ КЛАСС СЮДА В CHESS_NEURAL_NETWORK.PY ДЛЯ КЛИ-СКРИПТОВ:
class ChessTacticsResNet(nn.Module):
    def __init__(self, num_classes=27):
        super(ChessTacticsResNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ChessResBlock(64), ChessResBlock(64), ChessResBlock(64),
            ChessResBlock(64), ChessResBlock(64), ChessResBlock(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear((64 * 8 * 8) + 48, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, board_tensor, scalar_features):
        x = self.in_conv(board_tensor)
        x = self.res_blocks(x)
        x = torch.flatten(x, start_dim=1)
        combined_input = torch.cat((x, scalar_features), dim=1)
        out = self.classifier(combined_input)
        return out

# Ультимативная комбинированная модель CNN + LSTM по лекалам AUA 2026!
class ChessTacticsResNetLSTM(nn.Module):
    def __init__(self, num_classes=27, hidden_size=256):
        super(ChessTacticsResNetLSTM, self).__init__()

        # 1. Пространственный бэкбон (CNN + 6xResBlock)
        self.in_conv = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ChessResBlock(64), ChessResBlock(64), ChessResBlock(64),
            ChessResBlock(64), ChessResBlock(64), ChessResBlock(64)
        )

        # Разномерность выхода ResNet после Flatten: 64 канала * 8 * 8 = 4096 фич
        # Плюс приплюсовываем 48 скаляров правил = 4144 фичи на один временной шаг!
        self.cnn_features_dim = (64 * 8 * 8) + 48

        # 2. Временной контур памяти: Двухслойная LSTM сеть с постера AUA!
        self.lstm = nn.LSTM(
            input_size=self.cnn_features_dim,
            hidden_size=hidden_size,
            num_layers=2,  # Два последовательных слоя LSTM на схеме!
            batch_first=True
        )

        # 3. Финальный многоклассовый классификатор на полносвязных слоях
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Сигмоида для Multi-Label вероятностей
        )

    def forward(self, sequential_boards, sequential_scalars):
        # Входные тензоры имеют размерность последовательности:
        # board_tensor: [batch_size, sequence_length=5, 14, 8, 8]
        # scalar_tensor: [batch_size, sequence_length=5, 48]

        batch_size, seq_len, _, _, _ = sequential_boards.size()
        lstm_inputs = []

        # Прогоняем каждый временной шаг (кадр) независимо через ResNet
        for t in range(seq_len):
            board_t = sequential_boards[:, t, :, :, :]
            scalar_t = sequential_scalars[:, t, :]

            x = self.in_conv(board_t)
            x = self.res_blocks(x)
            x = torch.flatten(x, start_dim=1)

            # Конкатенируем свертки и правила для текущего шага
            combined_t = torch.cat((x, scalar_t), dim=1)
            lstm_inputs.append(combined_t.unsqueeze(1))

        # Склеиваем обратно во временную последовательность: [batch_size, 5, 4144]
        lstm_input = torch.cat(lstm_inputs, dim=1)

        # Пропускаем через LSTM контур
        lstm_out, (hn, cn) = self.lstm(lstm_input)

        # Берем скрытое состояние последнего временного шага (t=5) для классификации
        last_hidden_state = lstm_out[:, -1, :]

        out = self.classifier(last_hidden_state)
        return out


def run_diagnostic_lstm():
    print("🚀 [LSTM-INIT] Запуск теста ультимативной нейросети CNN+LSTM по схеме AUA...")

    # Симулируем батч из 2 наблюдений, где у каждого есть последовательность из 5 ходов
    mock_boards = torch.randn(2, 5, 14, 8, 8, dtype=torch.float32)
    mock_scalars = torch.randn(2, 5, 48, dtype=torch.float32)

    model = ChessTacticsResNetLSTM(num_classes=27)
    model.eval()

    with torch.no_grad():
        predictions = model(mock_boards, mock_scalars)

    print("\n🏁 [TEST-RESULTS] Спецификация ультимативного инференса:")
    print(f"📐 Форма результирующего тензора предсказаний: {predictions.shape} -> [2 партии, 27 классов тактики]")
    print(f"🔥 [SUCCESS] Слой временных зависимостей LSTM успешно интегрирован с ResNet-бэкбоном!")

    print("\n💾 Сохраняем финальную пуленепробиваемую архитектуру на SSD...")
    torch.save(model.state_dict(), "D:\\pythonprojects\\rechess_lstm_model.pth")
    print("🎉 [GLOBAL SUCCESS] Проект полностью соответствует научным лекалам Silicon Mountains!")


if __name__ == "__main__":
    run_diagnostic_lstm()
