from rag_pipeline import RAGPipeline
from make_rag_pipeline import make_pipeline

# TODO Auth layer
# TODO needs some form of Prompt Injection and guard rales
# TODO better chunking, look into chonkie
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
    # pipeline.index_file(r'_data/vtt/example_video_1.vtt')
    # pipeline.index_file(r'_data/vtt/example_video_2.vtt')
    # pipeline.index_file(r'_data/vtt/example_video_3.vtt')
    # pipeline.index_file(r'_data/vtt/example_video_4.vtt')
    # pipeline.index_file(r'_data/pdf/example_pdf_1.pdf')
    # pipeline.index_file(r'_data/mp3/example_mp3_1.mp3')
    # pipeline.index_file(r'_data/txt/example_txt_1.txt')
    # pipeline.index_file(r'_data/eml/example_email_1.eml')

def answer(query: str, pipeline: RAGPipeline):
    # print('-'*20)
    result = pipeline.answer_query(query)
    # print(f'Query: {query}')
    # print('Answer:', result['response'])
    # print('Source:', [m['source'] for m in result['matches']])
    return result

_pipeline = make_pipeline()

index_data(_pipeline)

# answer('list unhappy clients', pipeline)
# answer('tell me about mr bean\'s mortgage', pipeline)
result1 = answer('summarise recent life events of Miss Jones', _pipeline)
result2 = answer('What is Mr Smith unhappy about and what services can we provide to help', _pipeline)
result3 = answer('List names of clients who may be in need of mortgage or new wealth services and why they are of interest', _pipeline)
result5 = answer('List common themes of dissatisfaction our clients have recently expressed', _pipeline)

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

# example Qs
# ---------- 
# summarise recent life events of Miss Jones
# What is Mr Smith unhappy about and what services can evelyn partners provide to help
# List names of clients who may be in need of mortgage or new wealth services and why they are of interest
# List names of clients who have expressed dissatisfaction with us and why the clients are of interest
# List common themes of dissatisfaction our clients have recently expressed