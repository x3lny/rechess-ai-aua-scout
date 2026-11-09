import sys

sys.path.append('D:\\Lib\\site-packages')

import torch
import numpy as np
import board_renderer
import chess_neural_network
import feature_extractor

# Стерильный путь к весам базового ResNet для одиночных FEN-кадров
WEIGHTS_PATH = "D:\\pythonprojects\\rechess_resnet_model.pth"

# Полный, канонический список из 27 тактических параметров с баннера AUA!
THEMES_LIST = [
    'doubleCheck', 'pawnEndgame', 'rookEndgame', 'queenEndgame',
    'bishopEndgame', 'knightEndgame', 'promotion', 'advancedPawn',
    'mate', 'queenRookEndgame', 'sacrifice', 'exposedKing',
    'fork', 'attackingF2F7', 'skewer', 'capturingDefender',
    'discoveredAttack', 'opening', 'deflection', 'queensideAttack',
    'hangingPiece', 'pin', 'defensiveMove', 'kingsideAttack',
    'trappedPiece', 'quietMove', 'clearance'
]


def predict_tactical_weakness(fen_string):
    # 1. Формируем 14 каналов доски для одиночной позиции
    matrix = board_renderer.fen_to_tensor(fen_string)
    board_tensor = torch.from_numpy(matrix).unsqueeze(0)

    # 2. Извлекаем 48 позиционных скаляров правил
    scalar_features = feature_extractor.extract_aua_48_features(fen_string).unsqueeze(0)

    # 3. Вызываем класс ChessTacticsResNet, созданный для работы с одиночным инпутом
    model = chess_neural_network.ChessTacticsResNet(num_classes=len(THEMES_LIST))
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()

    with torch.no_grad():
        probabilities = model(board_tensor, scalar_features).squeeze(0).numpy()

    results = list(zip(THEMES_LIST, probabilities))
    results.sort(key=lambda x: x, reverse=True)

    print("\n🔮 [RECHESS-AI REPORT] Полные результаты сканирования тактических слабостей (Канонический спектр AUA):")
    print("-" * 75)
    # Хладнокровно выводим абсолютно все 27 параметров без ограничений и срезов!
    for theme, prob in results:
        status_icon = "💥" if prob * 100 > 30 else "🔹"
        print(f"{status_icon} Паттерн: {theme:<22} | Мощность уязвимости: {prob * 100:.1f}%")
    print("-" * 75)
