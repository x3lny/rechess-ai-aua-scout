import sys

# Твоя победная инъекция путей, чтобы Windows сразу видела пакеты!
sys.path.append('D:\\Lib\\site-packages')

import requests
import re
import scout_my_weakness


def fetch_and_scout_last_game(username="Jana_Gangsta"):
    print(f"⏳ [API-REQUEST] Подключение к серверам Chess.com... Сканирование профиля @{username}")

    headers = {'User-Agent': 'ReChess_AI_Bot (contact: chesscoder@example.com)'}

    # Прямой жесткий шлюз к логам твоих партий за ноябрь 2026 года!
    url_games = "https://api.chess.com/pub/player/jana_gangsta/games/2026/10"

    try:
        response = requests.get(url_games, headers=headers)

        if response.status_code != 200:
            print(f"🚨 [ERROR] Сервер отклонил запрос лога партий. Код: {response.status_code}")
            return

        games = response.json().get('games', [])

        if not games:
            print("🚨 [ERROR] В текущем месяце сыгранных партий не найдено.")
            return

        # Достаем самую последнюю сыгранную партию
        last_game = games[-1]

        # Извлекаем FEN финальной позиции
        final_fen = last_game.get('fen', None)

        if not final_fen:
            import re
            pgn = last_game.get('pgn', '')
            fen_match = re.search(r'\[FEN "([^"]+)"\]', pgn)
            if fen_match:
                final_fen = fen_match.group(1)

        if not final_fen:
            print("🚨 [ERROR] Не удалось оцифровать FEN-структуру из логов партии.")
            return

        print(f"🎯 [SUCCESS] Последняя партия аккаунта @{username} успешно оцифрована!")
        print(f"🧩 Финальный FEN позиции: {final_fen}\n")

        # Передаем FEN напрямую в твою обученную нейросеть!
        scout_my_weakness.predict_tactical_weakness(final_fen)

    except Exception as e:
        print(f"🚨 [SYSTEM CRASH] Непредвиденный сбой пайплайна: {str(e)}")


if __name__ == "__main__":
    # Запуск автоматического скаутинга твоих реальных уязвимостей на Chess.com!
    fetch_and_scout_last_game("Jana_Gangsta")
