import os
from typing import List, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import PyPDF2
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class GoogleDriveService:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Google Drive API 인증"""
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())
        
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def list_files(self, folder_id: str) -> List[Dict[str, Any]]:
        """폴더 내 파일 목록 조회"""
        query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='text/plain' or mimeType='application/vnd.google-apps.document')"
        
        results = self.service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        
        return results.get('files', [])
    
    def get_file_content(self, file_id: str, mime_type: str) -> str:
        """파일 내용 추출"""
        if mime_type == 'application/vnd.google-apps.document':
            return self._get_google_doc_content(file_id)
        elif mime_type == 'application/pdf':
            return self._get_pdf_content(file_id)
        elif mime_type == 'text/plain':
            return self._get_text_content(file_id)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {mime_type}")
    
    def _get_google_doc_content(self, file_id: str) -> str:
        """Google Docs 파일 내용 추출"""
        doc = self.service.files().export(
            fileId=file_id,
            mimeType='text/plain'
        ).execute()
        return doc.decode('utf-8')
    
    def _get_pdf_content(self, file_id: str) -> str:
        """PDF 파일 내용 추출"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        pdf_reader = PyPDF2.PdfReader(fh)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def _get_text_content(self, file_id: str) -> str:
        """텍스트 파일 내용 추출"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        return fh.read().decode('utf-8')
        
    def get_folder_contents(self, folder_id: str) -> List[Dict[str, Any]]:
        """폴더 내 파일 목록 조회 및 내용 추출"""
        try:
            logger.info(f"폴더 {folder_id}의 파일 목록 조회 시작")
            # 폴더 내 파일 목록 조회
            files = self.list_files(folder_id)
            if not files:
                logger.warning(f"폴더 {folder_id}에 파일이 없습니다.")
                return []
            
            logger.info(f"폴더 {folder_id}에서 {len(files)}개의 파일을 찾았습니다.")
            
            # 각 파일의 내용 추출
            documents = []
            for file in files:
                try:
                    logger.info(f"파일 내용 추출 중: {file['name']} (ID: {file['id']})")
                    content = self.get_file_content(file['id'], file['mimeType'])
                    documents.append({
                        'file_id': file['id'],
                        'file_name': file['name'],
                        'content': content
                    })
                    logger.info(f"파일 내용 추출 완료: {file['name']}")
                except Exception as e:
                    logger.error(f"파일 내용 추출 중 오류 발생: {file['name']} - {str(e)}")
                    continue
            
            logger.info(f"폴더 {folder_id}에서 총 {len(documents)}개의 문서 내용 추출 완료")
            return documents
        except Exception as e:
            logger.error(f"폴더 내용 조회 중 오류 발생: {str(e)}", exc_info=True)
            return [] 