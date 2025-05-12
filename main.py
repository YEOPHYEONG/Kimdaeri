import os
from dotenv import load_dotenv
import gradio as gr
from src.services.google_drive import GoogleDriveService
from src.services.cohere_service import CohereService
from src.core.report_generator import ReportGenerator

# 환경 변수 로드
load_dotenv()

# 서비스 초기화
drive_service = GoogleDriveService()
cohere_service = CohereService()
report_generator = ReportGenerator(drive_service, cohere_service)

def generate_report(folder_id: str, prompt: str) -> str:
    """
    보고서 생성 함수
    """
    try:
        # 폴더 ID와 프롬프트를 사용하여 보고서 생성
        report = report_generator.generate(folder_id, prompt)
        return report
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스 설정
with gr.Blocks(title="AI 김대리 - 드라이브 리포터") as demo:
    gr.Markdown("# AI 김대리 - 드라이브 리포터")
    gr.Markdown("Google Drive 폴더의 문서를 분석하여 보고서를 생성합니다.")
    
    with gr.Row():
        folder_id = gr.Textbox(
            label="Google Drive 폴더 ID",
            placeholder="폴더 ID를 입력하세요"
        )
    
    with gr.Row():
        prompt = gr.Textbox(
            label="보고서 생성 프롬프트",
            placeholder="원하는 보고서의 내용을 설명해주세요",
            lines=3
        )
    
    with gr.Row():
        generate_btn = gr.Button("보고서 생성")
    
    with gr.Row():
        output = gr.Textbox(
            label="생성된 보고서",
            lines=10
        )
    
    generate_btn.click(
        fn=generate_report,
        inputs=[folder_id, prompt],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch() 