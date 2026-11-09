import numpy as np


def fen_to_tensor(fen_string):
    """
    Трансформирует FEN-строку в трехмерный матричный тензор NumPy размера (14, 8, 8)
    в строгом соответствии с архитектурой CNN с Silicon Mountains 2026!
    """
    # Инициализируем пустой тензор 14 слоев, каждый 8х8 клеток
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    # Сплитуем FEN на позицию и метаданные
    parts = fen_string.split()
    position = parts[0]
    active_color = parts[1] if len(parts) > 1 else 'w'

    # Словари фиксации индексов слоев для сверточной сети
    piece_to_layer = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Белые фигуры (Слои 0-5)
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Черные фигуры (Слои 6-11)
    }

    # Разбиваем позицию на горизонтали (ranks)
    ranks = position.split('/')

    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                # Если цифра — пропускаем пустые поля двумерного массива
                file_idx += int(char)
            else:
                # Если фигура — ставим 1 в соответствующий слой матрицы
                if char in piece_to_layer:
                    layer_idx = piece_to_layer[char]
                    tensor[layer_idx, rank_idx, file_idx] = 1.0
                file_idx += 1

    # Заполняем Слой 12: Очередь хода (1.0 если ход белых, 0.0 если черных)
    if active_color == 'w':
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0

    # Слой 13: Базовая маска доски (вспомогательный слой геометрии)
    tensor[13, :, :] = 1.0

    return tensor


if __name__ == "__main__":
    # Тест-драбл: начальная позиция Ускоренного Дракона Богдана!
    test_fen = "r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5"

    print("⏳ [PROCESSING] Запуск матричного рендеринга тестовой FEN-строки...")
    matrix_out = fen_to_tensor(test_fen)

    print("🎯 [SUCCESS] Матрица успешно сгенерирована!")
    print(f"📐 Форма результирующего тензора: {matrix_out.shape} (Должна быть 14, 8, 8)")
    print(f"♟️ Проверка Слоя 0 (Белые пешки):\n{matrix_out[0]}")
    print(f"🚏 Проверка Слоя 12 (Ход белых):\n{matrix_out[12]}")

# Скопируй эту строку и вставь в самый низ файла в PyCharm
# Оставь это в самом низу файла на диске D
fen_to_tensor("r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5")
