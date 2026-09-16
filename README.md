# auto-novel-rag
An interactive novel generator powered by local LLM (MLX) and Wikipedia RAG. Features 2-stage XML plot building and infinite story loops. （ローカルLLMとWikipediaのRAGを組み合わせた、無限対話型の歴史小説ジェネレーター）


モデルのコンテキスト汚染やキャラクター崩壊を防ぐため、最新のLLMオーケストレーション手法（RAGによる史実バインドと、XMLタグを用いた2段階生成システム）で実装しています。

## 🚀 特徴 (Features)

- **完全自動スクレイピング職人**
  - DuckDuckGo Search API (`ddgs`) を介して指定した武将のWikipedia記事を自動検索。
  - `BeautifulSoup` を用いて、Wikipedia特有のノイズ（脚注、ナビゲーション、テーブル等）を徹底的にパース・除外します。
- **史実特化型RAG (Retrieval-Augmented Generation)**
  - `SentenceTransformer` と `FAISS` ベクトルデータベースを使用。
  - 物語の「導入」の雰囲気に合わせ、数ある史実データの中から最適な逸話や性格設定を抽出し、LLMのプロンプトに動的に注入します。
- **XMLプロット強制構築（2段階生成）**
  - 軽量モデル（3Bクラス）特有のハルシネーションを防ぐため、本文を直接書かせるのではなく、まず `<sho><ten><ketsu>` のXMLタグで物語の構造（プロット）を計算させます。
- **無限ループ執筆（対話型シミュレーション）**
  - AIが生成した本文に対し、ユーザーがテキストで次の展開（指示）を入力することで、TRPGやテキストベースのシミュレーションゲームのように無限に物語を紡ぎ続けることができます。

## 🛠 必須要件 (Prerequisites)

Apple Silicon (M1/M2/M3) 上での実行を前提として、`mlx-lm` を使用しています。

```bash
pip install mlx-lm requests beautifulsoup4 duckduckgo-search sentence-transformers faiss-cpu numpy
```

## ⚙️ 使い方 (Usage)

1. スクリプトを実行します。
```bash
python novel_generator.py
```
2. モードを選択します。史実に基づいたキャラクターを登場させたい場合は `2` を選択します。
3. 登場させたい武将の名前をスペース区切りで入力します。（例: `織田信長 徳川家康`）
4. 物語の「起」となる導入文を入力します。（例: `織田信長が突然、徳川家康に南蛮菓子の作り方を熱く語り始めた。`）
5. システムが自動でWikipediaを検索し、RAGデータベースを構築。その後、プロットの生成と第1段落の執筆が行われます。
6. 以降は、コンソールから続きの指示を入力することで物語が進みます。終了する場合は `End` と入力してください。

## ⚠️ 制限事項と動作の仕様 (Limitations)

- **LLMのスケール制限**: 3Bクラスのモデルを使用しているため、入力する武将の数や設定が多くなると、確率モデルの引力によって文脈が混線する（例：信長と家康の話に突然別人が乱入する）ことがあります。
- **著作権・ライセンス**: 情報ソースとしてWikipedia (CC BY-SA) に限定してスクレイピングを行うことで、生成物のクリーンさを保つ設計にしています。

## 📜 ライセンス (License)
MIT License
