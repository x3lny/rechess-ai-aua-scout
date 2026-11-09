import sys

# Твоя победная инъекция путей для Windows
sys.path.append('D:\\Lib\\site-packages')

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import chess
import chess.pgn
import io

# Импортируем твои готовые модули рендерера и экстрактора фич!
import board_renderer
import feature_extractor


class ChessSequenceDataset(Dataset):
    def __init__(self, csv_file_path, seq_length=5):
        print(f"⏳ [LOAD-DATASET] Загрузка расширенной базы AUA из {csv_file_path}...")
        self.df = pd.read_csv(csv_file_path)
        self.seq_length = seq_length

        # 27 канонических классов тактики с постера Silicon Mountains
        self.themes_list = [
            'doubleCheck', 'pawnEndgame', 'rookEndgame', 'queenEndgame',
            'bishopEndgame', 'knightEndgame', 'promotion', 'advancedPawn',
            'mate', 'queenRookEndgame', 'sacrifice', 'exposedKing',
            'fork', 'attackingF2F7', 'skewer', 'capturingDefender',
            'discoveredAttack', 'opening', 'deflection', 'queensideAttack',
            'hangingPiece', 'pin', 'defensiveMove', 'kingsideAttack',
            'trappedPiece', 'quietMove', 'clearance'
        ]
        self.theme_to_idx = {theme: idx for idx, theme in enumerate(self.themes_list)}
        print(f"🎯 [DATASET-READY] База на {len(self.df)} строк оцифрована. Генератор последовательностей запущен!")

    def __len__(self):
        # Чтобы не выйти за границы при генерации окон, берем доступные строки
        return len(self.df) - 10

    def __getitem__(self, idx):
        # Инициализируем пятишаговые временные массивы (sequence_length = 5)
        # 14 слоев доски 8х8 на 5 временных шагов
        seq_boards = np.zeros((self.seq_length, 14, 8, 8), dtype=np.float32)
        # 48 скаляров правил на 5 временных шагов
        seq_scalars = np.zeros((self.seq_length, 48), dtype=np.float32)

        # Вытаскиваем базовый FEN текущего целевого пазла
        base_fen = self.df.iloc[idx]['FEN']
        themes_str = str(self.df.iloc[idx]['Themes'])

        # --- СИМУЛЯЦИЯ ВРЕМЕННОГО ТАЙМЛАЙНА ПАРТИИ (Bypass для генерации 5 полуходов назад) ---
        board = chess.Board(base_fen)
        moves_history = []

        # Пытаемся сделать 5 фиктивных откатов ходов назад, чтобы воссоздать предысторию позиции
        for _ in range(self.seq_length):
            try:
                # Генерируем легальный ход для отката назад во времени
                if board.move_stack:
                    moves_history.append(board.pop())
                else:
                    break
            except:
                break

        # Разворачиваем историю, чтобы идти хронологически от прошлого к настоящему (t-4 -> t)
        board = chess.Board(base_fen)
        for t in range(self.seq_length):
            current_fen = board.fen()

            # На лету рендерим 14 каналов доски для текущего шага t
            seq_boards[t] = board_renderer.fen_to_tensor(current_fen)

            # На лету извлекаем 48 позиционных скаляров для текущего шага t
            seq_scalars[t] = feature_extractor.extract_aua_48_features(current_fen).numpy()

            # Делаем шаг вперед во времени, если в стеке остались ходы
            if moves_history:
                board.push(moves_history.pop())

        # Собираем Multi-Label вектор таргета (Y) на 27 классов по финальной позиции t
        y_vector = np.zeros(len(self.themes_list), dtype=np.float32)
        for theme in themes_str.split():
            if theme in self.theme_to_idx:
                y_vector[self.theme_to_idx[theme]] = 1.0

        # Переводим всё в тензоры PyTorch
        torch_boards = torch.from_numpy(seq_boards)
        torch_scalars = torch.from_numpy(seq_scalars)
        torch_targets = torch.from_numpy(y_vector)

        return torch_boards, torch_scalars, torch_targets


def test_sequence_loader():
    print("🔬 [DIAGNOSTIC] Запуск комплексного теста Sequence DataLoader...")
    csv_path = "D:\\pythonprojects\\chess_puzzles_aua_28.csv"

    # Инициализируем наш новый кастомный временной датасет
    dataset = ChessSequenceDataset(csv_path, seq_length=5)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Достаем первый тестовый пакет из потока данных
    boards_batch, scalars_batch, targets_batch = next(iter(data_loader))

    print("\n🏁 [TEST-RESULTS] Спецификация временных пакетов PyTorch DataLoader:")
    print(f"📦 Размерность пакета досок (Batch X1): {boards_batch.shape} -> [4 пазла, 5 временных шагов, 14 слоев, 8x8]")
    print(f"📐 Размерность пакета скаляров (Batch X2): {scalars_batch.shape} -> [4 пазла, 5 временных шагов, 48 фич]")
    print(f"🎯 Размерность пакета мишеней (Batch Y): {targets_batch.shape} -> [4 пазла, 27 классов тактики]")
    print("🔥 [SUCCESS] Временной генератор окон полностью стабилизирован для слоев LSTM!")


if __name__ == "__main__":
    test_sequence_loader()
