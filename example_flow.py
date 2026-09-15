from rag_pipeline import RAGPipeline
from make_rag_pipeline import make_pipeline
from pydantic_dataclasses import QueryRequest
from pydantic_dataclasses import DocumentScope

# TODO Maybe have a cross embedding for a prompt description and the query to pick the relevant prompt?
# TODO glossary + thesaurus for sparse search
# TODO Auth layer
# TODO needs some form of Prompt Injection and guard rales
# TODO VLM-based document parsing
# TODO k-graph
# TODO glossary 
# TODO maybe 'tools' so that the llm can ask for additional info that you would not expect to get from rag but would help the query 
# like:
# llm_tools = {
#     "get_market_data": get_market_data,
#     "get_portfolio": get_portfolio,
#     "get_products": get_products,
# }

# hf files in ~/.cache/huggingface/hub

def index_data(pipeline: RAGPipeline):
    pass
    # probably best to make a queue trigger that can process files in blob storage as prompted by the queue
    #pipeline.index_file(r'_data/vtt/example_video_1.vtt', client_reference=None)
    #pipeline.index_file(r'_data/vtt/example_video_2.vtt', client_reference=None)
    pipeline.index_file(r'_data/vtt/example_video_3.vtt', client_reference=123) # Mr Smith
    pipeline.index_file(r'_data/vtt/example_video_4.vtt', client_reference=456) # Ms Rose
    #pipeline.index_file(r'_data/pdf/example_pdf_1.pdf', client_reference=None)
    pipeline.index_file(r'_data/mp3/example_mp3_1.mp3', client_reference=789) # Mr Bean
    pipeline.index_file(r'_data/txt/example_txt_1.txt', client_reference=654) # Miss Jones
    pipeline.index_file(r'_data/eml/example_email_1.eml', client_reference=876) # Mr Thor

def answer(query: str, client_ref: int | None, scope: DocumentScope, pipeline: RAGPipeline):
    query_request = QueryRequest(
        query=query,
        client_reference=client_ref,
        scope=scope
    )
    result = pipeline.answer_query(query_request)
    return result

_pipeline = make_pipeline()

index_data(_pipeline)

# answer('list unhappy clients', pipeline)
# answer('tell me about mr bean\'s mortgage', pipeline)
result1 = answer('summarise recent life events of Miss Jones', 654, DocumentScope.CLIENT, _pipeline)
result2 = answer('What is Mr Smith unhappy about and what services can we provide to help', 123, DocumentScope.CLIENT, _pipeline)
result3 = answer('Name clients we can up-sell mortgage advice too due to debt or interest in buying property', None, DocumentScope.ALL_CLIENTS, _pipeline)
result5 = answer('List common themes of dissatisfaction our clients have recently expressed', None, DocumentScope.ALL_CLIENTS, _pipeline)
result6 = answer('Tell me about the attrition model', None, DocumentScope.INTERNAL, _pipeline)

print('*'*10)
print('*'*10)
print('*'*10)
print(result1['response'])
print('*'*10)
print(result2['response'])
print('*'*10)
print(result3['response'])
print('*'*10)
print(result5['response'])
print('*'*10)
print(result6['response'])
