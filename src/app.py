import gradio as gr
from services.cohere_service import CohereService
from services.google_drive import GoogleDriveService
import logging
import os # os 모듈 임포트 추가
from dotenv import load_dotenv # dotenv 임포트 추가
from typing import List, Dict, Any

# .env 파일 로드 (src 폴더의 부모 디렉토리에 있는 .env 파일을 명시적으로 지정)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # src 폴더 기준 상위 폴더의 .env
load_dotenv(dotenv_path=dotenv_path)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 서비스 초기화
cohere_service = CohereService()
drive_service = GoogleDriveService()

def process_folder_embeddings(folder_id: str, progress=gr.Progress()):
    """폴더의 문서들을 처리하여 임베딩 생성"""
    try:
        logger.info(f"Starting embedding process for folder: {folder_id}")
        
        # 구글 드라이브에서 문서 가져오기
        progress(0.2, desc="구글 드라이브에서 문서 가져오는 중...")
        documents = drive_service.get_folder_contents(folder_id)
        if not documents:
            return "문서를 찾을 수 없습니다."
        
        # 문서 처리 및 임베딩 생성
        progress(0.4, desc="문서 처리 중...")
        result = cohere_service.process_folder(folder_id, documents)
        
        if result:
            progress(1.0, desc="완료!")
            return f"임베딩이 성공적으로 생성되었습니다. 처리된 문서 수: {len(documents)}"
        else:
            return "임베딩 생성 중 오류가 발생했습니다."
            
    except Exception as e:
        logger.error(f"Error in process_folder_embeddings: {str(e)}", exc_info=True)
        return f"오류 발생: {str(e)}"

def generate_report_from_folder(folder_id: str, prompt: str, progress=gr.Progress()):
    """폴더의 문서들을 참고하여 보고서 생성"""
    try:
        logger.info(f"Generating report for folder: {folder_id}")
        progress(0.3, desc="관련 문서 검색 중...")
        
        # 보고서 생성
        progress(0.6, desc="보고서 생성 중...")
        report = cohere_service.generate_report(folder_id, prompt)
        
        progress(1.0, desc="완료!")
        return report
        
    except Exception as e:
        logger.error(f"Error in generate_report_from_folder: {str(e)}", exc_info=True)
        return f"오류 발생: {str(e)}"

def search_documents(folder_id: str, query: str, progress=gr.Progress()):
    """문서 검색"""
    try:
        logger.info(f"Searching documents in folder: {folder_id}")
        progress(0.3, desc="문서 검색 중...")
        
        # 문서 검색
        results = cohere_service.search_documents(folder_id, query)
        
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        
        # 결과 포맷팅
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            formatted_results.append(
                f"[{i}] 파일: {metadata['file_name']}\n"
                f"관련도 점수: {result['score']:.2f}\n"
                f"청크 인덱스: {metadata['chunk_index']}\n"
                f"파일 ID: {metadata['file_id']}\n"
                f"내용: {result['content']}\n"
            )
        
        progress(1.0, desc="완료!")
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}", exc_info=True)
        return f"오류 발생: {str(e)}"

# Gradio 인터페이스 구성
with gr.Blocks(title="AI 김대리 - 문서 분석 도우미") as demo:
    gr.Markdown("# AI 김대리 - 문서 분석 도우미")
    gr.Markdown("""
    **[사용법 안내]**
    1. **1번 탭에서 반드시 임베딩을 먼저 생성/누적**하세요. (동일 폴더ID로 여러 번 임베딩 가능)
    2. 임베딩이 완료된 후, 2/3번 탭에서 검색 및 보고서 생성을 진행하세요.
    3. 임베딩이 없는 폴더ID로 검색/생성 시 안내 메시지가 출력됩니다.
    """)
    with gr.Tabs():
        # 임베딩 생성 탭
        with gr.TabItem("1. 문서 임베딩 생성"):
            gr.Markdown("### 구글 드라이브 폴더의 문서들을 임베딩합니다. (누적 가능)")
            folder_id_input = gr.Textbox(
                label="구글 드라이브 폴더 ID",
                placeholder="폴더 ID를 입력하세요"
            )
            embed_button = gr.Button("임베딩 생성/누적 시작")
            embed_output = gr.Textbox(label="처리 결과")
            embed_button.click(
                fn=process_folder_embeddings,
                inputs=[folder_id_input],
                outputs=[embed_output]
            )
        # 문서 검색 탭
        with gr.TabItem("2. 문서 검색"):
            gr.Markdown("### 반드시 1번 탭에서 임베딩을 먼저 생성해야 합니다.")
            search_folder_id = gr.Textbox(
                label="구글 드라이브 폴더 ID",
                placeholder="폴더 ID를 입력하세요"
            )
            search_query = gr.Textbox(
                label="검색어",
                placeholder="검색어를 입력하세요"
            )
            search_button = gr.Button("검색")
            search_output = gr.Textbox(label="검색 결과")
            search_button.click(
                fn=search_documents,
                inputs=[search_folder_id, search_query],
                outputs=[search_output]
            )
        # 보고서 생성 탭
        with gr.TabItem("3. 보고서 생성"):
            gr.Markdown("### 반드시 1번 탭에서 임베딩을 먼저 생성해야 합니다.")
            report_folder_id = gr.Textbox(
                label="구글 드라이브 폴더 ID",
                placeholder="폴더 ID를 입력하세요"
            )
            report_prompt = gr.Textbox(
                label="보고서 요구사항",
                placeholder="보고서 요구사항을 입력하세요",
                lines=5
            )
            report_button = gr.Button("보고서 생성")
            report_output = gr.Textbox(label="생성된 보고서", lines=20)
            report_button.click(
                fn=generate_report_from_folder,
                inputs=[report_folder_id, report_prompt],
                outputs=[report_output]
            )

if __name__ == "__main__":
    demo.launch() 