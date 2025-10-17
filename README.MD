
AI Snake Trainer — Learn Reinforcement Learning the Fun Way
===========================================================

此專案示範如何用深度強化學習訓練一個「貪食蛇」(Snake) 智能體。程式碼包含多個環境變體、訓練/評估與簡易 UI 播放器。

專案重點
 - 使用 PyTorch (DQN 範例) 來訓練智能體
 - 使用 pygame 做簡單的遊戲視覺化與互動
 - 範例程式碼組織為多個章節（1..12），每個章節包含 5 個變體，並有訓練 / 測試 / UI 腳本

專案結構（重要檔案與資料夾）

根目錄
- dqn_snake.pth                      # 範例已訓練模型權重
- requirements.txt                   # 第三方套件列表（可用 pip install -r requirements.txt 安裝）
- README.MD                          # 本檔案

資料夾（每個資料夾內含數個範例檔案）
- 1/ .. 12/                          # 範例與練習章節
  - 1/ - 9/                          # 教學小節（環境、獎勵、狀態、動作、Q-Learning、Episode 等）
  - 10/ .. 12/                       # 進階範例（探索、multi env、awake、訓練/播放腳本）

範例檔案說明（局部示例）
- 12/train_dqn_snake.py              # DQN 訓練腳本
- 12/snake_play_dqn.py               # 使用訓練好的模型進行遊戲
- 12/snake_play_dqn_ui.py            # 帶 pygame UI 的播放介面

偵測到的第三方套件
- pygame
- torch
- numpy
- matplotlib

建立與執行（Windows / macOS）

先備條件
- 建議安裝 Python 3.10 或更新的 3.x 版本

macOS（使用 zsh 範例）
1) 建議使用 pyenv 或系統 Python。若使用 Homebrew 安裝 Python：

	brew install python

2) 建立虛擬環境並啟動：

	python3 -m venv .venv
	source .venv/bin/activate

3) 安裝依賴：

	pip install --upgrade pip
	pip install -r requirements.txt

4) 執行訓練或播放：

	python3 12/train_dqn_snake.py
	python3 12/snake_play_dqn_ui.py

Windows（PowerShell 範例）
1) 安裝 Python：請到 https://www.python.org/downloads/ 下載適合的安裝程式，建議勾選 "Add Python to PATH"。

2) 建立虛擬環境並啟動：

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

3) 安裝依賴：

	python -m pip install --upgrade pip
	pip install -r requirements.txt

4) 執行訓練或播放：

	python 12\train_dqn_snake.py
	python 12\snake_play_dqn_ui.py

注意（Mac / Windows）
- 如果使用 GPU 版的 PyTorch，請參考官方安裝指示（https://pytorch.org/get-started/locally/），根據你的 CUDA 版本選擇正確的安裝指令。
- 若碰到 pygame 無法在 headless 環境執行的問題，可設定 SDL_VIDEODRIVER 或在無 GUI 的情況下使用替代顯示方案。

如何貢獻
- 歡迎提交 PR：修正 bug、加入註解或新增訓練/視覺化範例。

其他說明
- 若要自訂 hyperparameters、環境設定，請查看各章節資料夾內的腳本並依註解修改。

聯絡
- 作者 / Maintainer: repo owner
