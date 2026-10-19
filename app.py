"""수업 3번(PDF RAG) + 수업 5번(LangGraph)을 연결한 Streamlit 앱."""
import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Streamlit 화면 설정은 첫 번째 Streamlit 명령이어야 합니다.
st.set_page_config(page_title="PDF · 웹 검색 챗봇", page_icon="📘")

from manual_rag import prepare_manual, answer_manual
from web_graph import build_web_graph

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")


def read_setting(name, default=""):
    """Secrets 파일이 있을 때만 읽고, 없으면 로컬 .env 환경 변수를 사용합니다."""
    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        BASE / ".streamlit" / "secrets.toml",
    ]
    if any(path.is_file() for path in secrets_paths):
        try:
            value = st.secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            # Secrets가 비어 있거나 형식이 올바르지 않으면 환경 변수로 대체
            pass
    return os.getenv(name, default).strip()


OPENAI_API_KEY = read_setting("OPENAI_API_KEY")
SERPAPI_KEY = read_setting("SERPAPI_KEY") or read_setting("SERPAPI_API_KEY")
MODEL = read_setting("OPENAI_MODEL", "gpt-4o-mini")

st.title("PDF · 웹 검색 챗봇")
st.caption("업로드한 PDF 또는 웹 검색 결과를 바탕으로 답변합니다.")

source = st.radio("검색 방식 선택", ["업로드한 PDF에서 찾기", "웹에서 찾기"], horizontal=True)

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("여기에_"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Secrets 또는 .env를 확인하세요.")
    st.stop()
if source == "웹에서 찾기" and (not SERPAPI_KEY or SERPAPI_KEY.startswith("여기에_")):
    st.error("SERPAPI_KEY가 설정되지 않았습니다. Secrets 또는 .env를 확인하세요.")
    st.stop()


# PDF 검색기는 사용자 세션별로 보관합니다. 전역 캐시는 사용하지 않습니다.
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
    st.session_state.pdf_index = None

if source == "업로드한 PDF에서 찾기":
    uploaded = st.file_uploader("PDF 파일 업로드", type=["pdf"], max_upload_size=20)
    if uploaded is None:
        st.session_state.pdf_hash = None
        st.session_state.pdf_index = None
        st.info("먼저 PDF 파일을 업로드하세요.")
    else:
        pdf_bytes = uploaded.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if file_hash != st.session_state.pdf_hash:
            st.session_state.pdf_index = None
            st.session_state.pdf_hash = file_hash
            try:
                with st.spinner("업로드한 PDF를 분석하고 있습니다..."):
                    st.session_state.pdf_index = prepare_manual(
                        OPENAI_API_KEY, MODEL, pdf_bytes
                    )
            except Exception as exc:
                st.error(f"PDF 준비 실패 ({type(exc).__name__}). 텍스트 PDF인지 확인하세요.")
        if st.session_state.pdf_index is not None:
            st.success(f"검색 준비 완료: {uploaded.name}")

@st.cache_resource(show_spinner=False)
def cached_graph(api_key, serpapi_key, model):
    return build_web_graph(api_key, serpapi_key, model)


# Streamlit은 새 입력마다 스크립트를 다시 실행하므로 대화 표시용 기록 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(f"[{item['source']}] {item['question']}")
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if item["sources"]:
            st.markdown(item["sources"])

question = st.chat_input("궁금한 내용을 질문하세요")
if question is not None:
    question = question.strip()
    if not question:
        st.warning("질문을 입력해 주세요.")
        st.stop()

    answer = ""
    sources = ""
    with st.spinner("답변을 만드는 중입니다..."):
        try:
            if source == "업로드한 PDF에서 찾기":
                if st.session_state.pdf_index is None:
                    st.warning("PDF 파일을 먼저 업로드해 주세요.")
                    st.stop()
                retriever, llm = st.session_state.pdf_index
                answer, pages = answer_manual(question, retriever, llm)
                if pages:
                    sources = "**검색된 참고 PDF 페이지:** " + ", ".join(
                        f"{page}페이지" for page in pages
                    )
            else:
                graph_app = cached_graph(OPENAI_API_KEY, SERPAPI_KEY, MODEL)
                result = graph_app.invoke({
                    "question": question, "search_results": [],
                    "searched_at": "", "answer": "", "error": "",
                })
                answer = result.get("answer", "답변이 없습니다.")
                # 실제 API가 제공한 제목/날짜/URL만 표시
                found = result.get("search_results", [])
                if found and not result.get("error"):
                    lines = ["**검색 자료 목록 (본문 미열람):**"]
                    for i, item in enumerate(found, 1):
                        title = item["title"].replace("[", "\\[").replace("]", "\\]")
                        date = item["date"] or "작성일 미제공"
                        lines.append(f"{i}. [{title}](<{item['link']}>) — {date}")
                    sources = "\n\n".join(lines)
        except FileNotFoundError:
            answer = "PDF 파일이 없습니다. 파일을 업로드해 주세요."
        except ValueError:
            answer = "PDF에서 텍스트를 읽을 수 없거나 설정값이 올바르지 않습니다."
        except Exception as exc:
            # 예외 원문에는 비밀값이나 요청 URL이 포함될 수 있으므로 유형만 노출
            answer = f"실행 오류({type(exc).__name__}). 패키지·API 키·서비스 상태를 확인하세요."

    st.session_state.chat_history.append({
        "source": source, "question": question, "answer": answer, "sources": sources,
    })
    st.rerun()
