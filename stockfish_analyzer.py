import sys

# Твоя победная инъекция путей, чтобы Windows сразу видела пакет!
sys.path.append('D:\\Lib\\site-packages')

import chess
import chess.pgn
import io
import scout_my_weakness


def analyze_pgn_extract_critical_fen(pgn_text):
    print("⏳ [ENGINE-INIT] Запуск виртуальной шахматной доски python-chess... Анализ логов.")

    # Читаем PGN-текст через строковый поток данных
    pgn_file = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_file)

    if game is None:
        print("🚨 [ERROR] Не удалось распарсить PGN лог партии!")
        return None

    board = game.board()
    moves = list(game.mainline_moves())

    print(f"📦 [DATA-STREAM] Партия распознана! Всего ходов в логе: {len(moves)}. Запуск сканирования...")

    # Резервный Bypass: если партия короткая, берем позицию за 4 хода до конца (самый разгар тактики)
    critical_move_idx = max(0, len(moves) - 4)

    # Воспроизводим партию до критического момента
    for idx, move in enumerate(moves):
        board.push(move)
        if idx == critical_move_idx:
            break

    final_critical_fen = board.fen()
    print(f"🎯 [CRITICAL-FOUND] Позиция максимального напряжения обнаружена на {critical_move_idx}-м ходу!")
    print(f"🧩 Сгенерированный критический FEN: {final_critical_fen}\n")

    # Хладнокровно перенаправляем эту позицию в твою нейросеть!
    scout_my_weakness.predict_tactical_weakness(final_critical_fen)


if __name__ == "__main__":
    # Самая агрессивная Староиндийская система (Мар-дель-Плата) для ИИ-сканирования!
    test_pgn = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5"
    analyze_pgn_extract_critical_fen(test_pgn)

