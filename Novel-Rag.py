import sys
import json
import os
import requests
import time
import urllib.parse
import gc
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from ddgs import DDGS
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

data_file = 'progress.json'

# =========================================================
# 完璧版・スクレイピング職人（Wikipedia & 戦国武将特化）
# =========================================================
def fetch_full_text(url, char_name=""):
    allowed_domains = ['ja.wikipedia.org']
    is_safe = any(domain in url for domain in allowed_domains)
    if not is_safe:
        return "" 

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Wikipediaの不要な構造（テーブル、ナビゲーション等）を事前削除
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "ul", "ol", "table"]):
            tag.decompose()

        # Wikipediaの本文エリアを特定
        body_area = soup.find('div', id='mw-content-text')
        if not body_area:
            body_area = soup.find('body')
            if not body_area:
                return ""

        clean_text = ""
        for p in body_area.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 20:
                # 史実・Wikipedia特有のノイズを排除
                if any(ng in text for ng in ["出典", "注釈", "関連項目", "外部リンク", "プライバシーポリシー", "ウィキペディア", "脚注", "参考文献"]):
                    continue
                clean_text += text + "\n"
        
        return clean_text
        
    except Exception as e:
        print(f"スクレイピング職人がエラーを吐きました: {e}")
        return ""


# =========================================================
# テキスト掃除＆特徴抽出の専用職人
# =========================================================
def clean_and_structure_text(raw_text):
    # 戦国武将の場合、Wikipediaからのセリフ抽出は難しい場合があるため、手紙や引用文としてのカギカッコを取得
    all_quotes = re.findall(r'「(.+?)」', raw_text)
    unique_quotes = list(dict.fromkeys(all_quotes))[:4]
    formatted_quotes = "\n".join([f"・「{q}」" for q in unique_quotes]) if unique_quotes else "（明記されたセリフ・引用は特になし）"
    
    cleaned_chunks = []
    for line in raw_text.split('\n'):
        if len(line) < 15:
            continue
        # 歴史系記事のノイズを除去
        if any(ng in line for ng in ["出典", "脚注", "リンク", "カテゴリ", "プロジェクト", "ポータル"]):
            continue
        # [1], [注 1] などのWikipediaの注釈記号を削除
        line_clean = re.sub(r'\[注釈?\s*\d+\]|\[\d+\]', '', line)
        if len(line_clean.strip()) > 10:
            cleaned_chunks.append(line_clean.strip())
            
    return formatted_quotes, cleaned_chunks


# =========================================================
# 完全汎用・検索 ＆ RAG関数
# =========================================================
def analyze_character_individually(char_name, embedder, first_input):
    # 戦国武将としてWikipediaを検索
    search_query = f'{char_name} 武将 site:ja.wikipedia.org'
    print(f"\n>>> 外部データベース（Web）から「{search_query}」を検索中...")
    
    char_raw_text = ""
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, region="jp-jp", max_results=10))
                
                for r in results:
                    url = r.get('href', '')
                    decoded_url = urllib.parse.unquote(url)
                    
                    if 'ja.wikipedia.org/wiki/' in decoded_url:
                        if "一覧" in decoded_url or "年表" in decoded_url:
                            continue
                        
                        print(f">>> {char_name} のデータ元 URL: {decoded_url}")
                        char_raw_text = fetch_full_text(url, char_name)
                            
                if char_raw_text:
                    break
                    
        except Exception as e:
            print(f"※ {char_name} の検索エラー (試行 {attempt+1}/3): {e}")
            time.sleep(3)

    if not char_raw_text:
        return ""

    print(f">>> 【データ成形】{char_name} の史実データ抽出とノイズ除去を実行中...")
    formatted_quotes, chunks = clean_and_structure_text(char_raw_text)

    print(f">>> 【RAG発動】{char_name} の性格・逸話に関連する文章を抽出中...")
    
    if chunks:
        chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)
        
        # ★史実の武将向けにプロンプトを調整
        query_text = f"物語の導入「{first_input}」という状況下における、{char_name}の史実に基づく行動原理、人間関係、性格や逸話"
        query_embedding = embedder.encode([query_text], normalize_embeddings=True)
        
        k = min(5, len(chunks))
        distances, indices = index.search(query_embedding, k)
        
        retrieved_setting = "\n".join([chunks[i] for i in indices[0]])
        del index
        
        structured_output = f"【引用・特徴的なセリフ】\n{formatted_quotes}\n\n【史実に基づく性格・特徴】\n{retrieved_setting}"
        return structured_output

    return ""

# =========================================================
# 1. 小説モードの起動とジャンル設定
# =========================================================
print("=" * 50)
print("=" * 50 + "\n")

print("モードを選択してください:")
print("1: オリジナル（時代小説など）")
print("2: 史実RAGモード（Wikipediaから戦国武将の史実を自動抽出）")

mode = ""
while mode not in ["1", "2"]:
    mode = input("選択 [1/2] (半角数字で入力): ").strip()
    mode = mode.replace("１", "1").replace("２", "2")

character_names = []
if mode == "2":
    while True:
        keyword = input("\n登場させたい戦国武将の名前をスペース区切りで入力してください（例: 織田信長 徳川家康）: ").strip()
        
        if keyword in ["", "1", "2", "22", "11"]:
            print("※ 警告：環境のバグで数字が混入しました。もう一度武将名を入力してください。")
            time.sleep(0.5)
            continue
            
        character_names = keyword.split()
        break 
    
    print(f"\n>>> 認識した武将: {character_names}")
    time.sleep(1)

print("\n【物語の導入（起）を入力してください】")
print("（例：織田信長が突然、徳川家康に南蛮菓子の作り方を熱く語り始めた。）")
first_input = ""
while not first_input:
    first_input = input("あなた [初回無料]: ").strip()
    if first_input in ["1", "2"]:
        print("※ 警告：環境のバグで数字が混入しました。物語の導入をもう一度入力してください。")
        first_input = ""
        time.sleep(0.5)

print(f"\n>>> 【確認】first_inputの中身はコレです -> 『{first_input}』")
time.sleep(2)

print("\n>>> 【システム】入力を受け付けました。これより全自動で史実検索・プロット構築・執筆を開始します...\n")

external_knowledge = ""
if mode == "2":
    combined_knowledge = []

    print(">>> 【システム】RAG用のEmbeddingモデルをロード中...")
    embedder = SentenceTransformer('intfloat/multilingual-e5-small')
    
    for char_name in character_names:
        char_setting = analyze_character_individually(char_name, embedder, first_input)
        if char_setting:
            combined_knowledge.append(f"【{char_name} の史実・設定】\n{char_setting}")
            print(f">>> {char_name} の史実抽出完了。\n")
        else:
            print(f">>> {char_name} のデータ取得に失敗しました。\n")
            
        print(">>> [システム] 連続アクセスを避けるため、3秒待機します...")
        time.sleep(3)

    del embedder
    gc.collect()
    print(">>> 【メモリ解放】全武将のRAG検索が完了したため、Embeddingモデルを破棄しました。")

    external_knowledge = "\n\n".join(combined_knowledge)

    debug_data = {"characters": combined_knowledge}
    with open('debug_settings.json', 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=4)
    print(">>> 【システム】取得した生データを debug_settings.json に保存しました！")
    
    if external_knowledge:
        print("=" * 40)
        print("【AIが認識した統合史実データ（生データ）】")
        print(external_knowledge)
        print("=" * 40 + "\n")
    else:
        print("【悲報】安全なサイト（Wikipedia等）から十分な史実が見つかりませんでした！")

# =========================================================
# 2. 自分専用AIのロード
# =========================================================
print(">>> 【システム】Ollamaのプロセスを強制終了してメモリを空けています...")
os.system("killall ollama")
time.sleep(3) 

print(">>> 【システム】ターミナルで錬成した俺専用モデル(4bit軽量版)をロード中...")
custom_model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

# =========================================================
# 2.5. MLX（Qwen）によるRAGデータの事前要約タスク
# =========================================================
summarized_knowledge = ""
if mode == "2" and combined_knowledge:
    print(">>> 【システム】MLX(Qwen)を使って、武将ごとに個別に史実設定を要約しています...")
    
    summarized_list = []
    
    for raw_char_data in combined_knowledge:
        char_title = raw_char_data.split('\n')[0]
        print(f">>> {char_title} のノイズを除去して要約中...")
        
        summary_system = f"""あなたは優秀な歴史小説の編集者です。
今回執筆する物語の導入は以下の通りです。
【導入】：{first_input}

【あなたのタスク】
提供された1人の武将の史実データから、上記の【導入】の雰囲気に適した『性格』と『逸話・口調』だけを抽出して要約してください。

【絶対命令】
1. 史実を曲げるような過度な脚色は避けつつ、小説に活かせる特徴を抽出すること。
2. 150文字以内で簡潔にまとめること。"""
        
        summary_messages = [
            {'role': 'system', 'content': summary_system},
            {'role': 'user', 'content': f"以下の武将の史実データを要約してください。\n\n{raw_char_data}"}
        ]
        
        summary_prompt = tokenizer.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
        summary_sampler = make_sampler(temp=0.1)
        
        summarized_result = generate(
            custom_model, tokenizer, 
            prompt=summary_prompt, 
            max_tokens=200, 
            sampler=summary_sampler,
            verbose=False
        )
        
        summarized_list.append(f"{char_title}\n{summarized_result.strip()}")

    summarized_knowledge = "\n\n".join(summarized_list)
    
    print("=" * 40)
    print("【MLXが抽出した高純度史実設定（個別処理版）】")
    print(summarized_knowledge)
    print("=" * 40 + "\n")

# =========================================================
# 2.8. 小説執筆AIのシステムプロンプト
# =========================================================
# 入力された武将名を動的にAIの指示に組み込む（なければ一般的な呼称）
chars_str = '、'.join(character_names) if character_names else '指定された武将'

system_content = f"""あなたはプロの歴史・時代小説作家です。
提供された史実設定と短い入力をもとに、日本の商業小説レベルの自然な本文を出力してください。

【AIへの絶対命令（厳守事項）】
1. 【登場人物】物語には主に {chars_str} を登場させ、フルネームの連呼を避け、自然な呼称（官途名、通称、または苗字など）で表記すること。
2. 【文体の指定】地の文は必ず「三人称視点」かつ「だ・である調（常体）」で書き、時代小説にふさわしい重厚感のある表現を心がけること。
3. 【過激描写の禁止】残酷な描写やグロテスクな展開は避けること。
4. 【フォーマット】会話は必ず「」で囲んで表現してください。
5. 【ループ禁止】絶対に同じ描写を繰り返さず、常に時間が進むようにしてください。"""

if summarized_knowledge:
    system_content += f"\n\n【公式・史実設定資料】\n{summarized_knowledge}"
messages = [{'role': 'system', 'content': system_content}]    

# =========================================================
# 3. ２段階生成システム（自動プロット構築 ＆ 初回の執筆）
# =========================================================
print("\n>>> 【システム】裏側でAIが『承・転・結』の展開（プロット）を計算中...\n")

plot_system = f"""あなたはプロの歴史小説シナリオライターです。
ユーザーが入力した【起】の展開とその【ジャンル・雰囲気】を完全に引き継ぎ、時間が前進するプロットを作成してください。

【最重要ルール】
1. キャラクターは {chars_str} を中心に展開すること。
2. 【起】で提示された状況を絶対に無視せず、戦国時代の背景や武将の性格を加味して展開させること。
3. AIのバグを防ぐため、各タグの中身は【絶対に1〜2文のみ】で簡潔に書くこと。

必ず以下のXMLタグフォーマットで出力してください。
<analysis>【起】の状況とジャンルをどう引き継ぐかの計算</analysis>
<sho>【起】に対する相手のリアクション</sho>
<ten>状況が変化する展開</ten>
<ketsu>物語のオチ</ketsu>"""

if summarized_knowledge:
    plot_system += f"\n\n【公式・史実設定資料】\n{summarized_knowledge}"

plot_messages = [
    {'role': 'system', 'content': plot_system},
    {'role': 'user', 'content': f'以下の出来事（起）から始まる物語の「起承転結」を作成してください。\n\n【起】：{first_input}'}
]

plot_prompt = tokenizer.apply_chat_template(plot_messages, tokenize=False, add_generation_prompt=True)

forced_plot_start = f"<analysis>\n【現在の状況】：{first_input}\n【絶対条件】：上記の状況から絶対に話を逸らさず、直接続きを描写する。\n"
plot_prompt += forced_plot_start  

plot_sampler = make_sampler(temp=0.6) 
plot_processors = make_logits_processors(repetition_penalty=1.15)
max_retries = 3
generated_plot = ""

for attempt in range(max_retries):
    print(f">>> 【システム】プロット生成中 (試行 {attempt + 1}/{max_retries})...")
    
    ai_reply = generate(
        custom_model, tokenizer, 
        prompt=plot_prompt, 
        max_tokens=800, 
        sampler=plot_sampler,
        logits_processors=plot_processors,
        verbose=False
    )
    full_ai_reply = forced_plot_start + ai_reply

    print("\n↓↓↓ 【デバッグ】AIの生出力ダンプ（パース前） ↓↓↓")
    print(full_ai_reply)
    print("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    try:
        sho_match = re.search(r'<sho>(.*?)</sho>', full_ai_reply, re.DOTALL)
        ten_match = re.search(r'<ten>(.*?)</ten>', full_ai_reply, re.DOTALL)
        ketsu_match = re.search(r'<ketsu>(.*?)</ketsu>', full_ai_reply, re.DOTALL)
        if sho_match and ten_match and ketsu_match:
            print(">>> 【システム】AIがXMLフォーマット通りに出力しました。パース成功！")
            generated_plot = f"【承】：{sho_match.group(1).strip()}\n【転】：{ten_match.group(1).strip()}\n【結】：{ketsu_match.group(1).strip()}"
            break
        else:
            print(">>> 【エラー】必要なタグが見つかりません。再計算させます...")
            del ai_reply
            gc.collect()
            
    except Exception as e:
        print(f">>> 【エラー】パース失敗: {e}")
        del ai_reply
        gc.collect()

if not generated_plot:
    print(">>> 【致命的エラー】AIが何度やってもフォーマットを守れませんでした。強制終了します。")
    sys.exit(1)

print("=" * 40)
print("【AIが自動生成した物語の軸（プロット）】")
print(generated_plot)
print("=" * 40 + "\n")

# --- Step 2: 本文の執筆（役者タスク） ---
print(">>> 【システム】完成したプロットを元に、AIが本文を執筆中...\n")

messages.append({
    'role': 'user', 
    'content': f'以下の【物語の軸（プロット）】に沿って、実際の時代小説の本文を書いてください。メタ発言や解説は厳禁です。\n\n【物語の軸】\n{generated_plot}'
})

novel_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
forced_start_text = f"{first_input}"
novel_prompt += forced_start_text

novel_sampler = make_sampler(temp=0.75)
novel_processors = make_logits_processors(repetition_penalty=1.05) 

ai_reply = generate(
    custom_model, tokenizer, 
    prompt=novel_prompt, 
    max_tokens=1000, 
    sampler=novel_sampler,
    logits_processors=novel_processors,
    verbose=False
)

full_novel_text = forced_start_text + ai_reply
print(f"【AI 時代小説パート】\n{full_novel_text}\n")
messages.append({'role': 'assistant', 'content': full_novel_text})

# =========================================================
# 4. 無限執筆ループ
# =========================================================
while True:
    print("-" * 40)
    user_input = input("あなた (終了は「End」): ")
    
    if user_input.strip().lower() == "end":
        print("\n物語をセーブしました。お疲れ様でした！")
        sys.exit(0)
        
    messages.append({
        'role': 'user', 
        'content': f'{user_input}\n\n（※AIとしての返事や質問は不要です。この展開に続く時代小説の本文のみを出力してください）'
    })
    
    if len(messages) > 5:
        messages = [messages[0]] + messages[-4:]
    
    print("\nAIが執筆中です...\n")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.75)
    processors = make_logits_processors(repetition_penalty=1.05) 
    
    ai_reply = generate(
        custom_model, tokenizer, 
        prompt=prompt, 
        max_tokens=1000, 
        sampler=sampler,
        logits_processors=processors,
        verbose=False
    )
    print(f"【AI 時代小説パート】\n{ai_reply}\n")
    messages.append({'role': 'assistant', 'content': ai_reply})
