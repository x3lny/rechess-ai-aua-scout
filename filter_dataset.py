import sys

# Твоя победная инъекция путей для Windows
sys.path.append('D:\\Lib\\site-packages')

import pandas as pd
import numpy as np


def compile_advanced_aua_dataset(input_csv_path, output_csv_path):
    print(f"⏳ [INIT] Запуск глубокой фильтрации датасета по лекалам AUA 2026...")
    print(f"📦 Чтение исходной базы {input_csv_path}...")

    # Читаем исходный скачанный датасет Lichess chunk-ами для экономии памяти Ryzen
    chunk_size = 100000
    chunks = []

    # Канонический список 28 тактических классов из таблицы Class-Level Accuracy постера AUA!
    aua_28_themes = [
        'doubleCheck', 'pawnEndgame', 'rookEndgame', 'queenEndgame',
        'bishopEndgame', 'knightEndgame', 'promotion', 'advancedPawn',
        'mate', 'queenRookEndgame', 'sacrifice', 'exposedKing',
        'fork', 'attackingF2F7', 'skewer', 'capturingDefender',
        'discoveredAttack', 'opening', 'deflection', 'queensideAttack',
        'hangingPiece', 'pin', 'defensiveMove', 'kingsideAttack',
        'trappedPiece', 'skewer', 'quietMove', 'clearance'
    ]

    # Превращаем в set для мгновенного поиска в памяти
    aua_themes_set = set(aua_28_themes)

    # Запускаем итератор по базе данных
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        # 1. Жесткая очистка: убираем строки без FEN или Themes
        chunk = chunk.dropna(subset=['FEN', 'Themes'])

        # 2. Быстрый Bypass-фильтр: оставляем только те пазлы, где есть хотя бы одна тема из 28 классов AUA
        def has_aua_theme(themes_str):
            parts = str(themes_str).split()
            return any(t in aua_themes_set for t in parts)

        mask = chunk['Themes'].apply(has_aua_theme)
        filtered_chunk = chunk[mask]

        chunks.append(filtered_chunk[['FEN', 'Themes']])

        # Мягко ограничиваем размер MVP для быстрой итерации на Asus TUF (целимся в 50 000 строк)
        total_rows = sum(len(c) for c in chunks)
        if total_rows >= 50000:
            print(f"📊 Достигнут тактический лимит MVP: {total_rows} строк.")
            break

    # Соединяем отфильтрованные блоки
    final_df = pd.concat(chunks, ignore_index=True)

    # Берем ровно 50 000 чистейших позиций для сплит-обучения
    final_df = final_df.head(50000)

    print(f"💾 Сохранение стерильного датасета AUA на SSD...")
    final_df.to_csv(output_csv_path, index=False)

    print("=" * 70)
    print(f"🎉 [SUCCESS] Новый датасет успешно сформирован!")
    print(f"📐 Итоговая размерность: {final_df.shape} (50 000 строк, 2 колонки)")
    print(f"🗂️ Физический путь: {output_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    # Запуск компиляции новой расширенной базы данных
    input_path = "D:\\pythonprojects\\lichess_db_puzzle.csv"
    output_path = "D:\\pythonprojects\\chess_puzzles_aua_28.csv"

    compile_advanced_aua_dataset(input_path, output_path)
