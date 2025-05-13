import os
import cohere
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import json
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CohereService:
    def __init__(self):
        """Cohere 서비스 초기화"""
        try:
            api_key = os.getenv('COHERE_API_KEY')
            if not api_key:
                raise ValueError("COHERE_API_KEY environment variable is not set")
            
            self.co = cohere.Client(api_key=api_key)
            self.index_dir = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'indices'))
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            # FAISS 인덱스 초기화
            self.dimension = 1024  # Cohere embed-multilingual-v3.0의 임베딩 차원
            self.index = None
            self.documents = []
            
            logger.info("CohereService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CohereService: {str(e)}", exc_info=True)
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            sentences = text.split('.')
            chunks = []
            current_chunk = []
            current_size = 0

            for sentence in sentences:
                sentence = sentence.strip() + '.'
                if current_size + len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_size += len(sentence)

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            logger.debug(f"Text chunked into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}", exc_info=True)
            raise

    def create_embeddings(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """텍스트 임베딩 생성"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            logger.debug(f"Text content preview: {[text[:100] + '...' for text in texts]}")
            
            response = self.co.embed(
                texts=texts,
                model='embed-multilingual-v3.0',
                input_type=input_type
            )
            
            embeddings = response.embeddings
            logger.info("Embeddings created successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            raise

    def save_index(self, folder_id: str):
        """FAISS 인덱스 저장"""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            # 인덱스 저장
            index_path = self.index_dir / f"{folder_id}.index"
            faiss.write_index(self.index, str(index_path))
            
            # 문서 메타데이터 저장
            metadata_path = self.index_dir / f"{folder_id}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Index and metadata saved for folder {folder_id}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            raise

    def load_index(self, folder_id: str) -> bool:
        """FAISS 인덱스 로드"""
        try:
            index_path = self.index_dir / f"{folder_id}.index"
            metadata_path = self.index_dir / f"{folder_id}_metadata.json"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.warning(f"No index found for folder {folder_id}")
                return False
            
            # 인덱스 로드
            self.index = faiss.read_index(str(index_path))
            
            # 메타데이터 로드
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f"Index and metadata loaded for folder {folder_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            raise

    def process_folder(self, folder_id: str, documents: List[Dict[str, Any]]):
        """폴더의 문서들을 처리하여 임베딩을 생성하고 저장"""
        try:
            logger.info(f"Processing {len(documents)} documents from folder {folder_id}")
            
            # 문서 청크 분할
            all_chunks = []
            chunk_metadata = []
            for doc in documents:
                chunks = self.chunk_text(doc['content'])
                all_chunks.extend(chunks)
                chunk_metadata.extend([{
                    'document_id': doc['file_id'],  # Google Drive의 file_id 사용
                    'chunk_id': i,
                    'content': chunk,
                    'metadata': {
                        'file_name': doc['file_name'],
                        'file_id': doc['file_id'],
                        'mime_type': doc.get('mime_type', ''),
                        'created_time': doc.get('created_time', ''),
                        'modified_time': doc.get('modified_time', '')
                    }
                } for i, chunk in enumerate(chunks)])
            
            # 청크 임베딩 생성
            embeddings = self.create_embeddings(all_chunks)
            embeddings_np = np.array(embeddings).astype('float32')
            
            # FAISS 인덱스 생성
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings_np)
            self.documents = chunk_metadata
            
            # 인덱스 저장
            self.save_index(folder_id)
            
            logger.info(f"Successfully processed folder {folder_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing folder: {str(e)}", exc_info=True)
            raise

    def search_documents(self, folder_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 검색 (RAG)"""
        try:
            logger.info(f"Searching documents in folder {folder_id} with query: {query}")
            
            # 인덱스 로드
            if not self.load_index(folder_id):
                logger.error(f"No index found for folder {folder_id}")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.create_embeddings([query], input_type="search_query")[0]
            
            # FAISS 검색
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), min(20, len(self.documents)))
            
            # 검색 결과 준비
            search_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    search_results.append({
                        'text': doc['content'],
                        'metadata': {
                            'file_name': doc['metadata']['file_name'],
                            'file_id': doc['metadata']['file_id'],
                            'chunk_index': doc['chunk_id'],  # chunk_id를 chunk_index로 매핑
                            'mime_type': doc['metadata'].get('mime_type', ''),
                            'created_time': doc['metadata'].get('created_time', ''),
                            'modified_time': doc['metadata'].get('modified_time', '')
                        },
                        'score': float(1 / (1 + distances[0][i]))  # 거리를 유사도 점수로 변환
                    })
            
            if not search_results:
                return []
            
            # Rerank 수행
            try:
                rerank_response = self.co.rerank(
                    query=query,
                    documents=search_results,
                    model='rerank-multilingual-v3.0',
                    top_n=min(top_k, len(search_results))
                )
                
                # 최종 결과 포맷팅
                results = []
                for result in rerank_response:
                    results.append({
                        'content': result.document['text'],
                        'score': result.relevance_score,
                        'metadata': result.document['metadata']  # 메타데이터 구조 유지
                    })
                
                logger.info(f"Found {len(results)} results after reranking")
                return results
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}", exc_info=True)
                # rerank 실패 시 원래 검색 결과 반환
                return [{
                    'content': doc['text'],
                    'score': doc['score'],
                    'metadata': doc['metadata']  # 메타데이터 구조 유지
                } for doc in search_results[:top_k]]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}", exc_info=True)
            raise

    def generate_queries(self, user_message: str) -> List[str]:
        """사용자 메시지에서 검색 쿼리 생성"""
        try:
            logger.info(f"Generating queries from user message: {user_message}")
            
            response = self.co.chat(
                message=f"""다음 사용자 메시지에서 문서 검색을 위한 2-3개의 검색 쿼리를 생성해주세요.
                각 쿼리는 사용자의 의도를 정확히 반영해야 합니다.
                쿼리는 쉼표로 구분된 리스트 형태로 반환해주세요.
                
                사용자 메시지: {user_message}""",
                temperature=0.3,
                max_tokens=100
            )
            
            # 응답에서 쿼리 추출
            queries = [q.strip() for q in response.text.split(',')]
            logger.info(f"Generated queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}", exc_info=True)
            return [user_message]  # 오류 발생 시 원본 메시지를 쿼리로 사용

    def calculate_similarity(self, a: List[float], b: List[float]) -> float:
        """두 임베딩 벡터 간의 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def generate_report(self, folder_id: str, prompt: str) -> str:
        """보고서 생성"""
        try:
            logger.info(f"Generating report for folder {folder_id}")
            
            # 검색 쿼리 생성
            queries = self.generate_queries(prompt)
            logger.info(f"Generated queries for report: {queries}")
            
            # 각 쿼리로 문서 검색
            all_relevant_docs = []
            for query in queries:
                docs = self.search_documents(folder_id, query, top_k=3)
                all_relevant_docs.extend(docs)
            
            # 중복 제거 및 점수 기준 정렬
            unique_docs = {}
            for doc in all_relevant_docs:
                key = f"{doc['metadata'].get('document_id', '')}_{doc['metadata'].get('chunk_id', '')}"
                if key not in unique_docs or doc['score'] > unique_docs[key]['score']:
                    unique_docs[key] = doc
            
            # 상위 문서 선택
            relevant_docs = sorted(unique_docs.values(), key=lambda x: x['score'], reverse=True)[:5]
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return "관련 문서를 찾을 수 없습니다."
            
            # 컨텍스트 구성
            context = "\n\n".join([
                f"[문서 {i+1}] {doc['content']}"
                for i, doc in enumerate(relevant_docs)
            ])
            
            # 프롬프트 구성
            full_prompt = f"""다음은 보고서를 작성하기 위한 컨텍스트와 요청사항입니다:

컨텍스트:
{context}

요청사항:
{prompt}

위 정보를 바탕으로 다음 형식에 맞춰 상세한 보고서를 작성해주세요:

# 보고서 제목

## 1. 개요
- 주요 내용 요약
- 핵심 포인트

## 2. 상세 분석
- 주요 내용 분석
- 중요 데이터 및 통계
- 인사이트

## 3. 결론 및 제언
- 주요 발견사항
- 향후 제언

## 4. 참고 자료
- 인용된 문서 및 출처

보고서는 한국어로 작성해주세요. 반드시 제공된 여러 문서를 골고루 참고하여 종합적인 보고서를 작성하세요."""

            # 보고서 생성
            response = self.co.chat(
                message=full_prompt,
                temperature=0.7,
                max_tokens=3000
            )
            
            logger.info("Report generated successfully")
            return response.text
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            raise 