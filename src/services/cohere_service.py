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
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            logger.error("COHERE_API_KEY environment variable is not set")
            raise ValueError("COHERE_API_KEY environment variable is not set")
        self.co = cohere.Client(api_key)
        self.embeddings_dir = Path("embeddings")
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # FAISS 인덱스 초기화
        self.dimension = 1024  # Cohere embed-multilingual-v3.0의 임베딩 차원
        self.index = None
        self.documents = []
        logger.info("CohereService initialized successfully")

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

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            logger.debug(f"Text content preview: {[text[:100] + '...' for text in texts]}")
            
            response = self.co.embed(
                texts=texts,
                model='embed-multilingual-v3.0',
                input_type='search_document'
            )
            
            logger.info("Embeddings created successfully")
            return response.embeddings
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
            index_path = self.embeddings_dir / f"{folder_id}.index"
            faiss.write_index(self.index, str(index_path))
            
            # 문서 메타데이터 저장
            metadata_path = self.embeddings_dir / f"{folder_id}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Index and metadata saved for folder {folder_id}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            raise

    def load_index(self, folder_id: str) -> bool:
        """FAISS 인덱스 로드"""
        try:
            index_path = self.embeddings_dir / f"{folder_id}.index"
            metadata_path = self.embeddings_dir / f"{folder_id}_metadata.json"
            
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
        """폴더의 문서들을 처리하여 임베딩을 누적 생성 및 저장"""
        try:
            logger.info(f"Processing {len(documents)} documents from folder {folder_id}")
            
            # 문서 청크 분할
            all_chunks = []
            chunk_metadata = []
            for doc in documents:
                chunks = self.chunk_text(doc['content'])
                all_chunks.extend(chunks)
                chunk_metadata.extend([{
                    'file_id': doc['file_id'],
                    'file_name': doc['file_name'],
                    'chunk_index': i,
                    'content': chunk
                } for i, chunk in enumerate(chunks)])
            
            # 청크 임베딩 생성
            embeddings = self.create_embeddings(all_chunks)
            embeddings_np = np.array(embeddings).astype('float32')
            
            # 기존 인덱스/메타데이터가 있으면 불러와서 누적, 없으면 새로 생성
            index_path = self.embeddings_dir / f"{folder_id}.index"
            metadata_path = self.embeddings_dir / f"{folder_id}_metadata.json"
            if index_path.exists() and metadata_path.exists():
                logger.info(f"기존 인덱스와 메타데이터를 불러와 누적합니다.")
                self.load_index(folder_id)
                self.index.add(embeddings_np)
                self.documents.extend(chunk_metadata)
            else:
                logger.info(f"새 인덱스와 메타데이터를 생성합니다.")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(embeddings_np)
                self.documents = chunk_metadata
            
            # 인덱스 저장
            self.save_index(folder_id)
            
            logger.info(f"Successfully processed folder {folder_id} (누적 저장)")
            return True
        except Exception as e:
            logger.error(f"Error processing folder: {str(e)}", exc_info=True)
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

    def search_documents(self, folder_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """문서 검색 (RAG)"""
        try:
            logger.info(f"Searching documents in folder {folder_id} for query: {query}")
            
            # 인덱스 로드
            if not self.load_index(folder_id):
                logger.error(f"No index found for folder {folder_id}")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.create_embeddings([query])[0]
            
            # FAISS로 초기 검색 (더 많은 결과 가져오기)
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'),
                min(20, len(self.documents))  # 최대 20개 결과 (더 다양한 문서 찾기)
            )
            
            # 검색 결과 준비
            search_results = []
            seen_document_keys = set()  # 중복 문서 식별용
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS가 결과를 찾지 못한 경우 -1 반환
                    # 파일ID와 청크 인덱스로 구성된 고유 키
                    doc_key = f"{self.documents[idx]['file_id']}_{self.documents[idx]['chunk_index']}"
                    
                    # 중복 문서 방지
                    if doc_key not in seen_document_keys:
                        seen_document_keys.add(doc_key)
                        search_results.append({
                            'text': self.documents[idx]['content'],
                            'metadata': {
                                'file_name': self.documents[idx]['file_name'],
                                'file_id': self.documents[idx]['file_id'],
                                'chunk_index': self.documents[idx]['chunk_index']
                            }
                        })
            
            if not search_results:
                return []
            
            logger.info(f"FAISS 초기 검색 결과: {len(search_results)}개 문서")
            logger.debug(f"FAISS 검색 문서 파일명: {[doc['metadata']['file_name'] for doc in search_results[:5]]}")
            
            # Cohere rerank로 재순위화
            try:
                rerank_response = self.co.rerank(
                    query=query,
                    documents=search_results,
                    model='rerank-multilingual-v3.0',
                    top_n=min(top_k, len(search_results))
                )
                
                # 최종 결과 포맷팅
                results = []
                seen_rerank_keys = set()  # rerank 결과에서 중복 방지
                
                for result in rerank_response:
                    # 파일ID와 청크 인덱스로 구성된 고유 키
                    result_key = f"{result.document['metadata']['file_id']}_{result.document['metadata']['chunk_index']}"
                    
                    # 중복 문서 방지
                    if result_key not in seen_rerank_keys:
                        seen_rerank_keys.add(result_key)
                        results.append({
                            'metadata': result.document['metadata'],
                            'score': result.relevance_score,
                            'content': result.document['text']
                        })
                
                logger.info(f"Rerank 후 결과: {len(results)}개 문서 (중복 제거됨)")
                logger.debug(f"Rerank 점수: {[r['score'] for r in results]}")
                logger.debug(f"Rerank 문서 파일명: {[r['metadata']['file_name'] for r in results]}")
                
                # 상위 top_k개 결과만 반환
                return results[:top_k]
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}", exc_info=True)
                # 실패 시 FAISS 결과 그대로 반환
                results = []
                seen_faiss_keys = set()  # 다시 중복 체크
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1:
                        # 파일ID와 청크 인덱스로 구성된 고유 키
                        doc_key = f"{self.documents[idx]['file_id']}_{self.documents[idx]['chunk_index']}"
                        
                        # 중복 문서 방지
                        if doc_key not in seen_faiss_keys and len(results) < top_k:
                            seen_faiss_keys.add(doc_key)
                            results.append({
                                'metadata': {
                                    'file_name': self.documents[idx]['file_name'],
                                    'file_id': self.documents[idx]['file_id'],
                                    'chunk_index': self.documents[idx]['chunk_index']
                                },
                                'score': 1.0 - distances[0][i] / (distances[0][0] if distances[0][0] > 0 else 1.0),
                                'content': self.documents[idx]['content']
                            })
                
                logger.info(f"Returning {len(results)} results from FAISS (중복 제거됨, rerank 없음)")
                return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}", exc_info=True)
            raise

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
                docs = self.search_documents(folder_id, query, top_k=3)  # 각 쿼리당 상위 3개 문서
                all_relevant_docs.extend(docs)
                
                # 검색된 문서 점수와 내용 미리보기 로깅
                for i, doc in enumerate(docs):
                    logger.debug(f"쿼리 '{query}' 결과 {i+1}. 파일: {doc['metadata']['file_name']}, 점수: {doc['score']:.6f}")
                    logger.debug(f"  내용 미리보기: {doc['content'][:100]}...")
            
            # 중복 제거 및 점수 기준 정렬
            unique_docs = {}
            for doc in all_relevant_docs:
                key = f"{doc['metadata']['file_id']}_{doc['metadata']['chunk_index']}"
                if key not in unique_docs or doc['score'] > unique_docs[key]['score']:
                    unique_docs[key] = doc
            
            # 최소 점수 임계값 설정 (최고 점수의 5%만 되어도 포함)
            max_score = max([doc['score'] for doc in unique_docs.values()], default=0)
            min_score_threshold = max_score * 0.05  # 최고 점수의 5%
            
            logger.info(f"최고 점수: {max_score:.6f}, 임계값: {min_score_threshold:.6f}")
            
            # 임계값 이상 문서만 선택하여 정렬
            filtered_docs = [doc for doc in unique_docs.values() if doc['score'] >= min_score_threshold]
            relevant_docs = sorted(filtered_docs, key=lambda x: x['score'], reverse=True)[:5]
            
            logger.info(f"보고서 작성을 위한 문서 {len(relevant_docs)}개 선택됨 (임계값: {min_score_threshold:.6f})")
            for i, doc in enumerate(relevant_docs):
                logger.info(f"참고문서 {i+1}. 파일: {doc['metadata']['file_name']}, 점수: {doc['score']:.6f}")
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return "관련 문서를 찾을 수 없습니다."
            
            # 전체 문서 내용 조회 (필요시)
            try:
                from src.services.google_drive import GoogleDriveService
                drive_service = GoogleDriveService()
                
                # 선택된 문서들의 파일 ID 수집
                file_ids = set()
                for doc in relevant_docs:
                    file_ids.add(doc['metadata']['file_id'])
                
                # 파일 전체 내용 조회
                full_contents = {}
                for file_id in file_ids:
                    # 폴더 내 모든 파일 조회
                    folder_contents = drive_service.get_folder_contents(folder_id)
                    
                    # 해당 파일 찾기
                    for content in folder_contents:
                        if content['file_id'] == file_id:
                            full_contents[file_id] = {
                                'file_name': content['file_name'],
                                'content': content['content']
                            }
                            break
                
                logger.info(f"전체 문서 {len(full_contents)}개 조회 완료")
            except Exception as e:
                logger.warning(f"전체 문서 조회 중 오류 발생, 청크 내용만 사용합니다: {str(e)}")
                full_contents = {}
            
            # 컨텍스트 구성 - 전체 문서와 관련 청크 모두 포함
            context_parts = []
            
            # 1. 전체 문서 내용 (있는 경우)
            for doc in relevant_docs:
                file_id = doc['metadata']['file_id']
                if file_id in full_contents:
                    context_parts.append(
                        f"[전체 문서] 파일: {full_contents[file_id]['file_name']}\n"
                        f"내용: {full_contents[file_id]['content']}"
                    )
                    # 이미 전체 문서를 포함했으므로 해당 파일 ID 제거
                    full_contents.pop(file_id, None)
            
            # 2. 관련 청크 (전체 문서가 없는 경우)
            for doc in relevant_docs:
                file_id = doc['metadata']['file_id']
                # 이미 전체 문서로 포함되지 않은 문서의 청크만 추가
                if file_id not in full_contents:
                    context_parts.append(
                        f"[청크] 파일: {doc['metadata']['file_name']}\n"
                        f"내용: {doc['content']}"
                    )
            
            # 최종 컨텍스트
            context = "\n\n".join(context_parts)
            
            logger.debug(f"Context preview: {context[:200]}...")
            logger.debug(f"Prompt: {prompt}")
            
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

            # chat 메서드 사용
            response = self.co.chat(
                message=full_prompt,
                temperature=0.7,
                max_tokens=3000,  # 토큰 수 증가 (더 긴 보고서 허용)
                p=0.75,
                k=0
            )
            
            logger.info("Report generated successfully")
            logger.debug(f"Generated report preview: {response.text[:200]}...")
            return response.text
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            raise 