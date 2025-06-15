# quizmaker

日本語のドキュメントからクイズを生成する Streamlit アプリです。

## 機能概要
- PDF, Word, Excel, PowerPoint, テキストファイルから内容を読み込み、5 択クイズを自動生成します。
- 「学習者モード」と「教育者モード」を切り替えられます。
  - 学習者モード: 画面上でクイズに回答し、その場で正誤判定と解説を確認できます。
  - 教育者モード: クイズを CSV 形式でダウンロードできます。過去に生成した CSV を読み込んで表示することも可能です。
- OpenAI API を利用して問題文や選択肢、解説を生成します。

## 事前準備
1. Python 3.9 以降をインストールしてください。
2. 必要なライブラリをインストールします。
   ```bash
   pip install streamlit langchain-openai langchain-community python-dotenv
   ```
3. OpenAI の API キーを取得し、環境変数 `OPENAI_API_KEY` に設定してください。
   `.env` ファイルに以下のように記載しても構いません。
   ```
   OPENAI_API_KEY=sk-...
   ```

## 使い方
1. リポジトリをクローンし、上記の準備を済ませます。
2. 以下のコマンドでアプリを起動します。
   ```bash
   streamlit run quizmaker_app.py
   ```
3. ブラウザが自動で開くので、ドキュメントをアップロードして「クイズを作成」を押してください。

## ライセンス
MIT License
