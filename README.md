# FinLora-RAG-Real-time-Financial-Risk-Analysis-with-LoRA-tuned-Llama-3-SEC-RAG
本项目通过LoRA微调Llama-3-8B与RAG实时检索SEC文件（Item 1A/7），构建企业级金融分析AI。采用BAAI/bge-large-en嵌入模型，经严格验证（困惑度↓/ROUGE↑/BLEU↑/BERTScore↑）与下游评估（事实准确性92.5%+），提供可溯源的精准决策API。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

> **精准分析10-K报告风险因素(Item 1A)与经营讨论(Item 7)**  
> 基于Llama-3-8B的LoRA微调与实时SEC检索增强技术，提供可溯源的金融决策支持

## 🚀 核心功能实现

### 1. LoRA微调训练 (`lora_finetune_train.py`)
```python
# 自定义评估类实现
class ModelEvaluator:
    def calculate_perplexity(self, dataset):
        """计算验证集困惑度"""
        self.model.eval()
        with torch.no_grad():
            # 批量计算损失
            total_loss = 0
            for batch in dataset:
                outputs = self.model(**batch, labels=batch.input_ids)
                total_loss += outputs.loss.item()
        return torch.exp(torch.tensor(total_loss / len(dataset)))

    def evaluate_generation_quality(self, dataset):
        """评估生成质量：ROUGE/BLEU/BERTScore"""
        # 随机采样生成预测
        predictions, references = [], []
        for item in sampled_dataset:
            prompt = format_prompt(item["question"], item["context"])
            generated = generate_response(prompt)
            predictions.append(generated)
            references.append(item["answer"])
        
        # 计算指标
        rouge_scores = self.rouge_scorer.score(references, predictions)
        bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
        bert_precision, bert_recall, bert_f1 = self.bert_scorer.score(predictions, references)
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "bert_f1": bert_f1.mean().item()
        }

# LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.2
)
# 训练流程
trainer = CustomSFTTrainer(
    model=model,
    train_dataset=financial_qa_dataset,
    eval_dataset=val_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        max_steps=60,
        eval_steps=10
    )
)
trainer.train()
```

### 2. RAG增强推理 (inference_RAG_main.py)
```python
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
```

### 📂 项目结构
```md
FinLora-RAG/
├── lora_finetune_train.py       # 模型微调训练脚本
├── inference_RAG_main.py        # RAG推理与对话脚本
├── content/
│   └── drive/
│       └── LLaMa3-Financial-Analyst/
│           └── Fine-l3_finagent_step60_eval/  # 预训练模型
├── requirements.txt             # Python依赖
└── README.md                    # 项目文档
```

### ⚙️ 快速开始
1. 环境安装
```bash
# 安装依赖
pip install -r requirements.txt

# 核心依赖：
# torch==2.3.0
# transformers==4.40.0
# unsloth==0.4.0
# sec-api==1.2.0
# faiss-cpu==1.8.0
# langchain==0.2.0
```

2. 训练金融领域模型
```bash
#请在训练前，将lora的参数以及评估函数的参数设定改为你本轮实验的值
python lora_finetune_train.py
```
训练好的模型将会自动保存至/content/drive/LLaMa3-Financial-Analyst/Fine-l3_finagent_step60_eval文件夹下

3.启动RAG对话系统
```bash
python inference_RAG_main.py

# 交互示例:
What Ticker Would you Like to Analyze? ex. AAPL: AAPL
----- 
Getting Filing Data...
----- 
Initializing Vector Database...
-----
Filing Initialized

What would you like to know about AAPL's form 10-K? 分析2023年供应链风险
```

4.示例回答
```bash
LLaMa3 Agent: According to the data, the net sales in the Americas increased by 3% in 2024 compared to 2023, primarily due to higher net sales of Services.
What would you like to know about AAPL's form 10-K? 
```


## 相关工具支持

- [swanlab](https://github.com/SwanHubX/SwanLab)：开源、现代化设计的深度学习训练跟踪与可视化工具
- [transformers](https://github.com/huggingface/transformers)：HuggingFace推出的包含预训练文本、计算机视觉、音频、视频和多模态模型的库，用于推理和训练
- [peft](https://github.com/huggingface/peft)：用于高效微调大型语言模型的库、
- [HuggingFace token](https://huggingface.co/settings/tokens)：用于导入相关数据集及模型
- [Free SEC API Key](https://sec-api.io/)：免费获取SEC EDGAR数据的API密钥 
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)：基础与训练模型
- [Supervised Fine-Tuning Trainer](https://huggingface.co/docs/trl/sft_trainer)：TRL库的监督微调训练器文档
- [Large English Embedding Model](https://huggingface.co/BAAI/bge-large-en-v1.5)：用于检索的嵌入模型
