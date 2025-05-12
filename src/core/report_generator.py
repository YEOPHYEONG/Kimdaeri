from typing import List, Dict, Any
import json
import os
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ReportGenerator:
    def __init__(self, drive_service, cohere_service):
        self.drive_service = drive_service
        self.cohere_service = cohere_service
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate(self, folder_id: str, prompt: str) -> str:
        """보고서 생성 메인 함수"""
        try:
            # 1. 폴더 내 파일 목록 조회
            files = self.drive_service.list_files(folder_id)
            
            # 2. 각 파일의 내용 추출
            documents = []
            for file in files:
                content = self.drive_service.get_file_content(file['id'], file['mimeType'])
                documents.append({
                    'content': content,
                    'file_name': file['name']
                })
            
            # 3. 문서 청킹
            chunks = self._chunk_documents(documents)
            
            # 4. 프롬프트 임베딩
            prompt_embedding = self.cohere_service.create_embeddings([prompt])[0]
            
            # 5. 문서 청크 임베딩
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = self.cohere_service.create_embeddings(chunk_texts)
            
            # 6. 유사도 기반 검색
            relevant_chunks = self._find_relevant_chunks(prompt_embedding, chunks, chunk_embeddings)
            
            # 7. 문서 재순위화
            reranked_chunks = self.cohere_service.rerank_documents(
                prompt,
                [chunk['text'] for chunk in relevant_chunks]
            )
            
            # 8. 보고서 생성
            context = [chunk['text'] for chunk in relevant_chunks]
            report = self.cohere_service.generate_report(prompt, context)
            
            return report
        except Exception as e:
            print(f"보고서 생성 중 오류 발생: {str(e)}")
            raise
    
    def _chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """문서를 청크로 분할"""
        chunks = []
        for doc in documents:
            text = doc['content']
            words = text.split()
            
            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i + chunk_size])
                chunks.append({
                    'text': chunk_text,
                    'file_name': doc['file_name']
                })
        
        return chunks
    
    def _find_relevant_chunks(self, query_embedding: List[float], chunks: List[Dict[str, Any]], chunk_embeddings: List[List[float]], top_k: int = 10) -> List[Dict[str, Any]]:
        """유사도 기반으로 관련 청크 검색"""
        try:
            # 코사인 유사도 계산
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            
            # 상위 K개 청크 선택
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [chunks[i] for i in top_indices]
        except Exception as e:
            print(f"관련 청크 검색 중 오류 발생: {str(e)}")
            raise 