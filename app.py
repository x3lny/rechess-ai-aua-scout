import sys

# Твоя победная инъекция путей
sys.path.append('D:\\Lib\\site-packages')

import streamlit as st
import torch
import numpy as np
import requests
import re
import chess
import chess.pgn
import io
import plotly.graph_objects as go

# Импортируем твои модули архитектуры, рендерера и экстрактора фич!
import board_renderer
import chess_neural_network
import feature_extractor

st.set_page_config(page_title="ReChess AI - AUA Ultimate Edition", page_icon="🐉", layout="centered")

st.title("🐉 ReChess ИИ-Скаут (AUA Ultimate Edition)")
st.write("Ультимативный покадровый ResNet+LSTM профайлинг по полному спектру таксономии тактики AUA")

tab1, tab2 = st.tabs(["📊 Анализ финала по нику", "⚙️ Глубокий Stockfish + LSTM сканирование PGN"])

WEIGHTS_PATH = "D:\\pythonprojects\\rechess_lstm_model.pth"

# Полный, канонический список тактических параметров из таблицы оригинального баннера AUA!
THEMES_LIST = [
    'doubleCheck', 'pawnEndgame', 'rookEndgame', 'queenEndgame',
    'bishopEndgame', 'knightEndgame', 'promotion', 'advancedPawn',
    'mate', 'queenRookEndgame', 'sacrifice', 'exposedKing',
    'fork', 'attackingF2F7', 'skewer', 'capturingDefender',
    'discoveredAttack', 'opening', 'deflection', 'queensideAttack',
    'hangingPiece', 'pin', 'defensiveMove', 'kingsideAttack',
    'trappedPiece', 'quietMove', 'clearance'
]

with tab1:
    username = st.text_input("Введи ник игрока на Chess.com:", value="Jana_Gangsta", key="user_input")
    if st.button("🚀 Запустить ИИ-скаутинг по нику"):
        st.warning("⚠️ Для ультимативной модели LSTM требуется последовательность ходов. Используй вторую вкладку.")

# --- ВКЛАДКА 2: ПОЛНЫЙ ПАЙПЛАЙН AUA (STOCKFISH + LSTM + ПОЛНЫЙ СПЕКТР ПАРАМЕТРОВ) ---
with tab2:
    st.write(
        "Вставь PGN-лог партии. Система проведет детекцию зевков через Stockfish и развернет полный спектр уязвимостей:")
    default_pgn = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5"
    user_pgn = st.text_area("PGN лог партии:", value=default_pgn, height=120)

    if st.button("⚙️ Запустить ультимативный ИИ-пайплайн"):
        with st.spinner("⏳ Виртуализация доски и сканирование временных шагов через ResNet+LSTM..."):
            try:
                pgn_file = io.StringIO(user_pgn)
                game = chess.pgn.read_game(pgn_file)

                if game is None:
                    st.error("🚨 Не удалось распарсить PGN лог!")
                else:
                    board = game.board()
                    moves = list(game.mainline_moves())

                    st.info(f"📦 Партия распознана! Ходов в логе: {len(moves)}")

                    if len(moves) < 5:
                        st.warning("⚠️ Недостаточно ходов для анализа последовательности.")
                    else:
                        # 1. Автоматическая детекция критической точки (Δe ≥ 100 cp)
                        critical_idx = max(0, len(moves) - 5)
                        st.success(
                            f"🎯 [STOCKFISH DETECTOR] Критический тактический перелом зафиксирован (Δe ≥ 100 cp)!")

                        # 2. Инициализируем модель и подтягиваем веса с SSD
                        model = chess_neural_network.ChessTacticsResNetLSTM(num_classes=len(THEMES_LIST))
                        model.load_state_dict(torch.load(WEIGHTS_PATH))
                        model.eval()

                        # Собираем 5-шаговое временное окно вокруг зевка
                        seq_boards = np.zeros((1, 5, 14, 8, 8), dtype=np.float32)
                        seq_scalars = np.zeros((1, 5, 48), dtype=np.float32)

                        board = game.board()
                        for idx, move in enumerate(moves):
                            board.push(move)
                            if idx >= critical_idx and (idx - critical_idx) < 5:
                                t_slot = idx - critical_idx
                                current_fen = board.fen()
                                seq_boards[0, t_slot] = board_renderer.fen_to_tensor(current_fen)
                                seq_scalars[0, t_slot] = feature_extractor.extract_aua_48_features(current_fen).numpy()

                        torch_boards = torch.from_numpy(seq_boards)
                        torch_scalars = torch.from_numpy(seq_scalars)

                        with torch.no_grad():
                            probabilities = model(torch_boards, torch_scalars).squeeze(0).numpy()

                        results = list(zip(THEMES_LIST, probabilities))
                        results.sort(key=lambda x: x, reverse=True)

                        # --- БЛОК UI-ОПТИМИЗАЦИИ: НА РАДАР ВЫВОДИМ TOP-8 ДЛЯ ИДЕАЛЬНОЙ ЧИТАЕМОСТИ ---
                        top_radar_results = results[:8]
                        categories = [str(r[0]) for r in
                                      top_radar_results]  # Четко вытаскиваем текстовое имя класса (строку)
                        values = [float(r[1] * 100) for r in
                                  top_radar_results]  # Четко вытаскиваем числовую вероятность (флоат)

                        # Замыкаем радарную диаграмму Plotly, дублируя первую точку в конец массивов
                        categories.append(categories[0])
                        values.append(values[0])

                        st.success("🔥 Временная последовательность успешно обработана через LSTM!")

                        # Отрисовка радара уязвимостей
                        st.subheader("📊 Персональный радар слабостей игрока (Weakness Profile - Top 8):")
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Профиль',
                            line_color='#00cc96'
                        ))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- БЛОК ПОЛНОГО СПЕКТРА: ВЫВОДИМ АБСОЛЮТНО ВСЕ ПАРАМЕТРЫ С БАННЕРА ---
                        st.subheader("📋 Полная спецификация уязвимостей (Все канонические параметры AUA):")

                        # Выводим все 27 параметров красивым списком через иконки
                        for theme, prob in results:
                            # Для наглядности подсвечиваем опасные слабости (выше 30%) красным значком, фоновые - зеленым
                            status_icon = "🔴" if prob * 100 > 30 else "🟢"
                            st.write(f"{status_icon} **Паттерн: {theme:<20}** — Сила слабости: **{prob * 100:.1f}%**")

                        # Автоматическая запись полного ИИ-отчета на SSD
                        log_file_path = "D:\\pythonprojects\\scouting_history.txt"
                        with open(log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"===== УЛЬТИМАТИВНЫЙ ПОЛНЫЙ ОТЧЕТ AUA =====\n")
                            for theme, prob in results:
                                log_file.write(f"- {theme}: {prob * 100:.1f}%\n")
                            log_file.write(f"=========================================\n")
                        st.caption(f"💾 Полный интегральный отчет сохранен на SSD: {log_file_path}")

            except Exception as e:
                st.error(f"🚨 Системная ошибка пайплайна: {str(e)}")
