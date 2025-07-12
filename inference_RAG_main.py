from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import logging
from unsloth import is_bfloat16_supported
import warnings

# 导入Pipeline & RAG 相关包
from sec_api import ExtractorApi, QueryApi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
logging.set_verbosity_error()

# HuggingFace TOKEN
hf_token = "your_hf_token"
# SEC-API Key
sec_api_key = "your_sec_api_key"

 #  重新定义提示词，加载微调模型
ft_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is a user question, paired with retrieved context. Write a response that appropriately answers the question,
include specific details in your response. <|eot_id|>

<|start_header_id|>user<|end_header_id|>

### Question:
{}

### Context:
{}

<|eot_id|>

### Response: <|start_header_id|>assistant<|end_header_id|>
{}"""

if True: 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/workspace/content/drive/LLaMa3-Financial-Analyst/Fine-l3_finagent_step60_eval", # 保存模型的位置
        max_seq_length = 2048, 
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)


# 定义推理函数，处理生成和解码令牌。
def inference(question, context):
  FastLanguageModel.for_inference(model)
  inputs = tokenizer(
  [
      ft_prompt.format(
          question,
          context,
          "", # output - leave this blank for generation!
      )
  ], return_tensors = "pt").to("cuda")

  # 参数通过重用先前计算的值来实现更快的生成。
  # 将‘ pad_token_id ’设置为EOS令牌以正确处理填充。
  outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True, pad_token_id=tokenizer.eos_token_id)
  response = tokenizer.batch_decode(outputs) # 将符号解码成英语单词
  return response

# 从完整的响应中提取语言模型生成的函数
def extract_response(text):
    text = text[0]
    start_token = "### Response: <|start_header_id|>assistant<|end_header_id|>"
    end_token = "<|eot_id|>"

    start_index = text.find(start_token) + len(start_token)
    end_index = text.find(end_token, start_index)

    if start_index == -1 or end_index == -1:
        return None

    return text[start_index:end_index].strip()


"""  

现在我们已经有了经过微调的语言模型、推理函数和所需的提示格式，我们现在需要设置RAG管道，以便将相关上下文注入到每一代中。
流程如下：
*用户问题* -> *上下文检索从10-K* -> *LLM回答用户问题使用上下文*
要做到这一点，我们需要能够：
1. 从10-K表格中收集具体信息
2. 解析并分块其中的文本
3. 矢量化和嵌入块到一个矢量数据库
4. 设置一个检索器，在数据库中对用户的问题进行语义搜索，以返回相关的上下文
表格10-K是美国证券交易委员会要求的年度报告，它提供了公司财务业绩的全面总结。

为了更容易地做到这一点，我们利用了SEC API 
对于这个项目，我们将专注于只加载**1A**和**7**
- **1A**：风险因素
- **7**：管理层对财务状况和经营结果的讨论和分析

"""


# 设置SEC 10-K数据管道和检索功能
# 提取文件功能
def get_filings(ticker):
    global sec_api_key

    # 使用QueryAPI查找最近的文件
    queryApi = QueryApi(api_key=sec_api_key)
    query = {
      "query": f"ticker:{ticker} AND formType:\"10-K\"",
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" } }]
    }
    filings = queryApi.get_filings(query)

    # 获取10-K URL
    filing_url = filings["filings"][0]["linkToFilingDetails"]

    # 使用ExtractorAPI提取文本
    extractorApi = ExtractorApi(api_key=sec_api_key)
    onea_text = extractorApi.get_section(filing_url, "1A", "text") # Section 1A - Risk Factors
    seven_text = extractorApi.get_section(filing_url, "7", "text") # Section 7 - Management’s Discussion and Analysis of Financial Condition and Results of Operations

    # 融合文本
    combined_text = onea_text + "\n\n" + seven_text

    return combined_text


# HF Model 路径
modelPath = "BAAI/bge-large-en-v1.5"
# 创建一个带有模型配置选项的字典，指定使用cuda进行GPU优化
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}

# 用指定的参数初始化一个LangChain的HuggingFaceEmbeddings实例
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # 提供预训练模型的路径
    model_kwargs=model_kwargs, # 传递模型配置选项
    encode_kwargs=encode_kwargs # 传递编码选项
)


"""
接下来需要处理和定义向量数据库
在这个流程中，我们从上述定义的 SEC API 函数获取数据，然后经历三个步骤：1. 文本分割2. 向量化3. 检索功能设置

文本分割是将大型文档或文本数据分解为更小、更易于管理的部分的过程。在处理诸如法律文件、财务报告或任何长篇文章等大量文本数据时，这通常是必要的。文本分割的目的是确保数据能够被机器学习模型和数据库有效地处理、分析和索引。

向量数据库以向量的形式存储数据，向量是文本、图像或其他类型数据的数值表示。这些向量能够捕捉数据的语义含义，从而实现高效的相似性搜索和检索。

我们这里使用的向量数据库是 [Facebook AI 语义搜索](https://ai.meta.com/tools/faiss/) 库，这是一个轻量级且内存中的解决方案（无需将其保存到磁盘），虽然不像其他向量数据库那样强大，但对于此用例来说效果很好。

如何将拆分文档和嵌入技术结合使用：
1. 嵌入： 当文本数据被分割成较小的片段时，每个片段都会通过嵌入模型转换为数值向量（嵌入）。这些嵌入能够捕捉文本的语义关系和含义。
2. 存储： 向量数据库会将这些嵌入向量与原始文本块的引用一同存储。
3. 索引： 数据库会对向量进行索引，以便实现快速高效的相似性搜索。这一索引过程会将向量以一种易于快速查找相似向量的方式进行组织。
4. 用法： 当进行查询时，向量数据库会搜索与查询向量最相似的向量（文本片段），根据它们的语义相似度检索出相关的文本片段。
"""
# 提示用户输入他们想要分析的股票行情
ticker = input("What Ticker Would you Like to Analyze? ex. AAPL: ")

print("-----")
print("Getting Filing Data")
# 检索指定代码的归档数据
filing_data = get_filings(ticker)

print("-----")
print("Initializing Vector Database")
# 初始化文本分割器，将文件数据分割成chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,         # 每个chunk的最大大小
    chunk_overlap = 500,       # chunks之间重叠的字符数
    length_function = len,     # 这是用来确定chunks的长度
    is_separator_regex = False # 分隔符是否为正则表达式模式
)
# 将归档数据分成更小的、易于管理的数据chunks
split_data = text_splitter.create_documents([filing_data])

# 使用嵌入从分割数据创建FAISS矢量数据库
db = FAISS.from_documents(split_data, embeddings)

# 创建要在矢量数据库中搜索的检索对象
retriever = db.as_retriever()

print("-----")
print("Filing Initialized")


"""
Retrieval是查询矢量数据库以查找并返回与给定查询匹配的相关文本块或文档的过程。这涉及到搜索索引嵌入，以识别与查询最相似的嵌入。
在这个函数中，查询用于调用Retrieval，Retrieval返回一个文档列表。然后提取这些文档的内容并作为查询的上下文返回。
**工作原理：**
1. **查询嵌入：**当查询生成时，首先使用与文本块相同的嵌入模型将其转换为嵌入。
2. **相似搜索：**检索器在向量数据库中搜索与查询嵌入相似的嵌入。这种相似性通常使用余弦相似度或欧几里得距离等距离度量来衡量。
3. **文档检索：**检索器然后检索与类似嵌入相关的原始文本块或文档。
4. **上下文组装：**检索到的文本块被组装以提供一个连贯的上下文或查询的答案。
"""
# Retrieval函数
def retrieve_context(query):
    global retriever
    retrieved_docs = retriever.invoke(query) # Invoke the retriever with the query to get relevant documents
    context = []
    for doc in retrieved_docs:
        context.append(doc.page_content) # Collect the content of each retrieved document
    return context

#然后把所有内容串到一个while循环中，该循环将接收用户的问题，从填充了特定公司表单10-K的Vector DB中检索上下文，然后通过我们的微调模型运行推断以生成响应
while True:
  question = input(f"What would you like to know about {ticker}'s form 10-K? ")
  if question == "x":
    break
  else:
    context = retrieve_context(question) # Context Retrieval
    resp = inference(question, context) # Running Inference
    parsed_response = extract_response(resp) # Parsing Response
    print(f"LLaMa3 Agent: {parsed_response}")
    print("-----\n")
