# 問診AI

streamlitによる直感的なUI作成
![メニュー画面](images/menu.png)

## 概要

## 主な機能

### 1. マルチLLMプロバイダー対応karma 

- vLLM（ローカルLLMサーバー - Qwen 2.5 72B）
- OpenAI（GPT-4 Turbo）
- Google Gemini（Gemini 2.0 Flash Lite）
- Anthropic Claude（Claude 3 Opus）
- Deepseek（Deepseek Chat）

### 2. Webインターフェース (`app.py`)

- Streamlitベースの使いやすいUI

### 3. 必要なパッケージのインストール

```bash
pip install -r requirements.txt
pip install google-generativeai
pip install openai
```

## 使用方法

### A. Webインターフェースでの実行（推奨）

1. Streamlitアプリケーションを起動：

    ```bash
    streamlit run app.py
    ```

2. ブラウザで以下のURLにアクセス：

    ```html
    http://localhost:8501
    ```

3. Webインターフェースの使用手順：

### B. コマンドライン実行　開発環境ではここを実行しないと上手くいかない。（対策は後で考える）

git config --global user.name "Motoya Takashi"
git config --global user.email "<XXX@gmail.com>"

## データ形式

## 制限事項
