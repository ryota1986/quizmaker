import tempfile
import os
import json
import time
import csv
import io
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@contextmanager
def temporary_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

def load_documents(uploaded_file) -> List[Document]:
    with temporary_file(uploaded_file) as tmp_path:
        suffix = Path(uploaded_file.name).suffix.lower()
        file_name = uploaded_file.name
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(tmp_path, mode="elements")
                docs = loader.load()
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif suffix == ".pptx":
                loader = UnstructuredPowerPointLoader(tmp_path)
            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding='utf-8')
            else:
                raise ValueError(f"サポートされていないファイル形式: {suffix}")
            documents = loader.load()
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['source'] = file_name
            return documents
        except Exception as e:
            raise Exception(f"ファイル '{file_name}' のドキュメント読み込みに失敗しました: {str(e)}")

def create_vector_store(documents: List[Document]) -> FAISS:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            texts.extend(chunks)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"ベクトルストアの作成に失敗しました: {e}")
        return None

def parse_quiz_json(quiz_text: str) -> List[Dict[str, Any]]:
    try:
        if "```json" in quiz_text:
            json_part = quiz_text.split("```json")[1].split("```")[0]
        elif "```" in quiz_text:
            json_part = quiz_text.split("```")[1].split("```")[0]
        else:
            json_part = quiz_text
        quizzes = json.loads(json_part.strip())
        if not isinstance(quizzes, list):
            quizzes = [quizzes]
        validated_quizzes = []
        for quiz in quizzes:
            if all(key in quiz for key in ["question", "options", "answer"]):
                if isinstance(quiz["options"], list) and len(quiz["options"]) >= 2:
                    validated_quizzes.append(quiz)
        return validated_quizzes
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        st.error(f"クイズのJSON解析に失敗しました: {e}")
        return []

def create_educator_csv_export(quizzes: List[Dict], filename: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["問題", "回答", "解説"])
    for quiz in quizzes:
        writer.writerow([
            quiz.get("question", ""),
            quiz.get("answer", ""),
            quiz.get("explanation", "")
        ])
    return ('\ufeff' + output.getvalue()).encode("utf-8")

def download_quiz_data(quizzes: List[Dict], filename: str):
    csv_bytes = create_educator_csv_export(quizzes, filename)
    st.download_button(
        label="📝 クイズCSVダウンロード",
        data=csv_bytes,
        file_name=f"quiz_{filename}_{int(time.time())}.csv",
        mime="text/csv"
    )

def display_quiz_results():
    if 'quiz_results' not in st.session_state or not st.session_state.quiz_results:
        return
    total_questions = len(st.session_state.quiz_results)
    correct_answers = sum(st.session_state.quiz_results.values())
    score_percentage = (correct_answers / total_questions) * 100
    st.markdown("---")
    st.markdown("### 📊 クイズ結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総問題数", total_questions)
    with col2:
        st.metric("正解数", correct_answers)
    with col3:
        st.metric("正解率", f"{score_percentage:.1f}%")
    if score_percentage >= 80:
        st.success("🎉 素晴らしい！よく理解できています！")
    elif score_percentage >= 60:
        st.info("👍 良い理解度です！")
    else:
        st.warning("📚 もう少し復習してみましょう！")

def main():
    st.set_page_config(page_title="ドキュメントクイズ生成アプリ", page_icon="📚", layout="wide")
    st.title("📚 ドキュメントクイズ生成アプリ")
    st.markdown("""PDF, Excel, Word, PowerPoint, テキストファイルからクイズを自動生成します。""")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が設定されていません")
        st.stop()

    if 'generated_quizzes' not in st.session_state:
        st.session_state.generated_quizzes = None
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = {}

    with st.sidebar:
        st.header("⚙️ 設定")
        app_mode = st.selectbox("モード選択", ["学習者モード", "教育者モード"])
        model_choice = st.selectbox("使用するモデル", ["gpt-4.1", "gpt-4.1-mini"])
        num_questions = st.slider("クイズ問題数", 5, 30, 10)
        difficulty = st.selectbox("難易度", ["基本", "標準", "応用"], index=1)
        difficulty_prompt = {
            "基本": "基本的な内容の理解を確認する",
            "標準": "重要な概念や詳細の理解を測る",
            "応用": "応用的な思考や分析を要求する"
        }[difficulty]

    if app_mode == "教育者モード":
        st.info("🧑‍🏫 教育者モードでは、クイズは画面に表示されず、CSVとしてダウンロードされます。")
        uploaded_csv = st.file_uploader("以前生成したクイズCSVをアップロードして表示する", type=["csv"])
        if uploaded_csv:
            quizzes = []
            reader = csv.DictReader(io.StringIO(uploaded_csv.read().decode("utf-8-sig")))
            for row in reader:
                if all(k in row for k in ("問題", "回答", "解説")):
                    quizzes.append({
                        "question": row["問題"],
                        "answer": row["回答"],
                        "explanation": row["解説"],
                        "options": [row["回答"], "ダミー1", "ダミー2", "ダミー3", "ダミー4"]
                    })
            if quizzes:
                st.success("✅ アップロードしたクイズを表示します")
                for idx, quiz in enumerate(quizzes, start=1):
                    st.markdown(f"**Q{idx}. {quiz['question']}**")
                    selected = st.radio(
                        "選択してください:", quiz['options'], key=f"csv_quiz_{idx}", index=None
                    )
                    if selected is not None:
                        is_correct = selected == quiz['answer']
                        st.session_state.quiz_results[f"csv_quiz_{idx}"] = is_correct
                        if is_correct:
                            st.success("✅ 正解！")
                        else:
                            st.error(f"❌ 不正解。正解は: {quiz['answer']}")
                        st.info(f"📖 解説: {quiz['explanation']}")
                display_quiz_results()

    st.markdown("### 📁 ファイルアップロード")
    uploaded_files = st.file_uploader(
        "ドキュメントを選択してください",
        type=["pdf", "xlsx", "xls", "docx", "pptx", "txt"],
        accept_multiple_files=True
    )

    quiz_chain = LLMChain(
        llm=ChatOpenAI(
            model_name=model_choice,
            temperature=0.3,
            max_tokens=4000,
        ),
        prompt=PromptTemplate.from_template("""
        以下はドキュメントの本文です：
        {document_text}

        この内容に基づき、{difficulty_prompt}5択クイズを{num_questions}問作成してください。

        要件：
        - 各問題は文書の重要なポイントに基づいて作成
        - 選択肢は5つ、1つだけが正解
        - 明確で分かりやすい質問文
        - 各問題に必ず日本語で詳細な解説を含める（なぜ正解なのかを根拠となる情報も交えて）

        以下のJSON形式で回答してください：
        [
        {{
            "question": "質問文",
            "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4", "選択肢5"],
            "answer": "正解の選択肢",
            "explanation": "正解の理由や参考文書の根拠を含めて日本語で解説"
        }}
        ]

        JSONのみを返してください。
        """)
    )

    if uploaded_files:
        if st.button("🚀 クイズを作成"):
            with st.spinner("ドキュメント処理中..."):
                all_docs = []
                for file in uploaded_files:
                    try:
                        all_docs.extend(load_documents(file))
                    except Exception as e:
                        st.error(f"{file.name} の読み込みに失敗しました: {e}")

                if all_docs:
                    full_text = "\n".join(doc.page_content for doc in all_docs)
                    response = quiz_chain.run(
                        document_text=full_text,
                        difficulty_prompt=difficulty_prompt,
                        num_questions=num_questions
                    )
                    st.session_state.generated_quizzes = parse_quiz_json(response)

    if st.session_state.generated_quizzes:
        quizzes = st.session_state.generated_quizzes
        if app_mode == "教育者モード":
            st.success("✅ クイズを生成しました（教育者モード）")
            download_quiz_data(quizzes, "quiz")
        else:
            st.success("✅ クイズを生成しました（学習者モード）")
            for idx, quiz in enumerate(quizzes, start=1):
                st.markdown(f"**Q{idx}. {quiz['question']}**")
                selected = st.radio(
                    "選択してください:", quiz['options'], key=f"quiz_{idx}", index=None
                )
                if selected is not None:
                    is_correct = selected == quiz['answer']
                    st.session_state.quiz_results[f"quiz_{idx}"] = is_correct
                    if is_correct:
                        st.success("✅ 正解！")
                    else:
                        st.error(f"❌ 不正解。正解は: {quiz['answer']}")
                    st.info(f"📖 解説: {quiz['explanation']}")
                st.markdown("---")
            display_quiz_results()

    else:
        st.info("👆 ドキュメントをアップロードして、\"クイズを作成\"を押してください")

if __name__ == "__main__":
    main()
