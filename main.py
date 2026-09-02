from rag_pipeline import RAGPipeline
from extractors import BaseDocumentExtractor, PDFExtractor, VTTExtractor, MP3Extractor, HTMLExtractor
from storage import SQLiteMetadataStore, UsearchVectorStore
from embedding import BGEEmbeddingService
from llm_caller import LocalQwenLLMCaller


pdf_extractor = PDFExtractor()
vtt_extractor = VTTExtractor()
mp3_extractor = MP3Extractor()
html_extractor = HTMLExtractor()

extractors: list[BaseDocumentExtractor] = [pdf_extractor, vtt_extractor, mp3_extractor, html_extractor]

embedding_service = BGEEmbeddingService()

sql_store = SQLiteMetadataStore("_database/rag_vectors.db")
vector_store = UsearchVectorStore("_database/vector_index.usearch", embedding_dim=embedding_service.embedding_dim)

llm_caller = LocalQwenLLMCaller()

pipeline = RAGPipeline(
    extractors=extractors,
    metadata_store=sql_store,
    vector_store=vector_store,
    encoder=embedding_service,
    llm_caller=llm_caller
)

# probably best to make a queue trigger that can process files in blob storage as prompted by the queue
pipeline.index_file(r'_data/vtt/example_video_1.vtt')
pipeline.index_file(r'_data/vtt/example_video_2.vtt')
pipeline.index_file(r'_data/pdf/example_pdf_1.pdf')
pipeline.index_file(r'_data/mp3/example_mp3_1.mp3')

# TODO needs some form of Prompt Injection and guard rales

# probably best to make an https endpoint that takes a query in the request
print('-'*20)
query = 'tell me about mr bean\'s mortgage'
result = pipeline.answer_query(query)
print(f'Query: {query}')
print('Answer:', result['response'])
print('Source:', [m['source'] for m in result['matches']])
