# HomaBot 매뉴얼·웹 검색 통합 안내 챗봇

## 수업 파일과 연결
- `manual_rag.py` : 3번 Simple RAG 수업의 PyPDFLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → FAISS → ChatOpenAI 활용. 단, **페이지 번호를 유지**하기 위해 PDF 페이지별 문서를 합치지 않습니다.
- `web_graph.py` : 5번 LangGraph 수업의 SerpAPI → web_search 노드 → generate_answer 노드 → StateGraph 활용.
- `app.py` : Streamlit 사용자가 검색 방식과 질문을 입력하고, 결과와 출처를 볼 수 있도록 연결.

## 실행 전 준비
1. 이 프로젝트 ZIP에는 사용자가 제공한 **`homabot_manual.pdf`**(3페이지)가 이미 포함되어 있습니다. 추가로 복사할 필요가 없습니다.
2. Jupyter에서 작업할 경우 `%pip install -r requirements.txt`를 실행합니다. Anaconda Prompt에서는 `python -m pip install -r requirements.txt`를 실행할 수 있습니다.
3. `.streamlit/secrets.toml.example`을 복사하여 `.streamlit/secrets.toml`을 만들고 본인 키를 입력하세요. 또는 작업 폴더의 `.env` 파일에 `OPENAI_API_KEY`, `SERPAPI_KEY`, `OPENAI_MODEL`을 설정해도 됩니다.
4. 터미널에서 `python -m streamlit run app.py` 실행. Windows에서 Jupyter와 같은 Python을 쓰려면 `"C:\Users\ASUS\anaconda3\python.exe" -m streamlit run app.py`처럼 실행할 수 있습니다.
5. 두 모드를 각각 질문하고 답변과 출처를 확인하세요.

## 종합실습 필수 테스트
- PDF에 있는 내용 질문: 응답과 `n페이지` 표시
- PDF에 없는 사양 질문: 매뉴얼에 없다는 안내 (모델 응답을 직접 확인)
- 웹 정보 질문: 검색 답변 + 원본 URL/날짜 표시
- 빈 질문: API 미호출
- SerpAPI 오류 또는 검색 결과 없음: 별도로 안내
- 키 누락: 키를 화면에 드러내지 않고 설정 안내

## 배포
1. `app.py`, `manual_rag.py`, `web_graph.py`, `requirements.txt`, `homabot_manual.pdf`, `.gitignore`를 GitHub에 올립니다. 공개 권한이 있는 PDF인지 확인하세요.
2. `.env`, `.streamlit/secrets.toml` 및 실제 API 키는 절대로 올리지 않습니다.
3. Streamlit Community Cloud에서 저장소/브랜치/app.py 선택, **Secrets**에 동일한 키를 등록해 배포합니다.
4. 배포한 URL에서 매뉴얼/웹 두 경로를 각각 실제 테스트합니다.

## 제공된 PDF의 내용에 기반한 실습 질문 (실제 API 답변은 별도 검증)

| 질문 | 매뉴얼에서 확인할 핵심 내용 | 원문 페이지 |
| --- | --- | --- |
| HomaBot은 처음 어떻게 설정하나요? | 뒷면 전원 3초 이상, 언어·시간대, 2.4 GHz Wi-Fi, 앱 등록 | 1 |
| SOS 버튼은 어떻게 쓰나요? | 상단 노란색 버튼을 2초 이상 누름 | 1 |
| 반려동물이 있는 집에서 자가 청소는 몇 개월마다 하나요? | Self-Clean Mode 권장 1개월 | 2 |
| 배터리 잔량 20% 미만이면 어떻게 되나요? | 도킹 스테이션으로 복귀해 충전 | 2 |
| 고객지원 전화번호는 무엇인가요? | 080-123-4567, 평일 09:00~18:00 (문서의 실습용 정보) | 3 |
| HX-PRO 카메라 해상도는 얼마인가요? | 구체적인 해상도가 기록되어 있지 않음 | 없음 |

> **확인할 사항:** 1페이지 표에는 HX-PRO의 기능으로 '공기질 관리'가 기재되어 있지만, 2페이지 3.2 절 제목은 'HX-2 공기질 모니터링'입니다. PDF 자체의 표기가 서로 다르므로 어느 모델의 정확한 사양인지 단정하지 않습니다.
>
> 문서 메타데이터의 작성일은 2031-07-15로 현재 날짜보다 미래입니다. **실습용 가상 매뉴얼**로 취급하고 실제 출시 제품 정보를 확인한 문서로 취급하지 않습니다.

## 주의
- PDF는 로컬에서 3페이지와 텍스트 추출 여부를 검증했습니다. 실제 임베딩 생성과 LLM 답변은 API 키 없이는 검증하지 못합니다.
- PDF 임베딩과 OpenAI 답변 생성, SerpAPI 검색에는 비용과 사용량이 발생할 수 있습니다.
- 웹 검색은 Google 검색 제목·요약·링크만 사용하며 링크 본문을 읽지는 않습니다.
- 화면에는 검색된 참고 페이지를 표시합니다. 해당 페이지가 답변을 모두 증명한다는 뜻은 아닙니다.
