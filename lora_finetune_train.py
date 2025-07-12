import logging
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
import atexit
import warnings
import sys

# 忽略特定的警告 - 修正版本
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Exception ignored in: <function tqdm.__del__")


# 微调相关包
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments,logging
from unsloth import is_bfloat16_supported

# 导入Pipeline & RAG 相关包
import swanlab
import time  # 用于性能计时
import gc

# 导入评估指标相关包
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
logging.set_verbosity_error()

# 创建一个安全的SwanLab包装器　
class SwanLabWrapper:
    def __init__(self):
        self.is_initialized = False
        self.run = None

    def init(self, **kwargs):
        try:
            self.run = swanlab.init(**kwargs)
            self.is_initialized = True
            return self.run
        except Exception as e:
            print(f"SwanLab 初始化警告: {e}")
            self.is_initialized = False
            return None

    def log(self, data):
        if self.is_initialized:
            try:
                swanlab.log(data)
            except Exception as e:
                print(f"SwanLab 日志警告: {e}")

    def finish(self):
        if self.is_initialized:
            try:
                swanlab.finish()
            except Exception as e:
                print(f"SwanLab 结束警告: {e}")
            finally:
                self.is_initialized = False


# 使用包装器
swanlab_wrapper = SwanLabWrapper()

# 初始化SwanLab
swanlab_wrapper.init(
    project="Financial_Analyst",
    workspace="flhxch",
    logdir="/workspace/content/saved_swanlab_logs",
    config={
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_seq_length": 2048,
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-5,
        "lora_dropout":0.1,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 60,
    }
)


# 注册退出处理函数
def cleanup_resources():
    """在程序退出时清理资源"""
    print("\n清理资源中...")
    try:
        # 先关闭SwanLab
        swanlab_wrapper.finish()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 强制垃圾回收
        gc.collect()

    except Exception as e:
        print(f"清理时出现警告: {e}")


atexit.register(cleanup_resources)

# Load the model and tokenizer from the pre-trained FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    token=hf_token,
)

# Apply LoRA (Low-Rank Adaptation) adapters to the model for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.2,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Defining the expected prompt 定义预期的提示符
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
{}<|eot_id|>"""

# Grabbing end of sentence special token 抓取句尾特殊标记
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


# Function for formatting above prompt with information from Financial QA dataset
def formatting_prompts_func(examples):
    questions = examples["question"]
    contexts = examples["context"]
    responses = examples["answer"]
    texts = []
    for question, context, response in zip(questions, contexts, responses):
        text = ft_prompt.format(question, context, response) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


# 加载数据集并划分训练集和验证集
full_dataset = load_dataset("virattt/llama-3-8b-financialQA", split="train")
full_dataset = full_dataset.map(formatting_prompts_func, batched=True, )

# 划分数据集：90%训练，10%验证
train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")


# 创建评估指标类
class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def calculate_perplexity(self, dataset, batch_size=4):
        """计算模型在数据集上的困惑度"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_texts = dataset[i:i + batch_size]['text']
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                total_loss += loss.item() * inputs.input_ids.shape[0]
                total_tokens += inputs.input_ids.shape[0]

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()

    def evaluate_generation_quality(self, dataset, num_samples=50):
        """评估生成质量：ROUGE、BLEU、BERTScore"""
        self.model.eval()
        FastLanguageModel.for_inference(self.model)

        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        bert_scores = {'precision': [], 'recall': [], 'f1': []}

        # 随机采样评估
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        for idx in indices:
            item = dataset[int(idx)]

            # 解析问题、上下文和真实答案
            text_parts = item['text'].split("### Response: <|start_header_id|>assistant<|end_header_id|>")
            if len(text_parts) < 2:
                continue

            prompt_part = text_parts[0] + "### Response: <|start_header_id|>assistant<|end_header_id|>"
            true_answer = text_parts[1].replace(EOS_TOKEN, "").strip()

            # 生成预测答案
            inputs = self.tokenizer([prompt_part], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response: <|start_header_id|>assistant<|end_header_id|>" in generated_text:
                pred_answer = generated_text.split("### Response: <|start_header_id|>assistant<|end_header_id|>")[
                    -1].strip()
            else:
                pred_answer = generated_text.strip()

            # 计算ROUGE分数
            rouge_result = self.rouge_scorer.score(true_answer, pred_answer)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_result[key].fmeasure)

            # 计算BLEU分数
            reference = word_tokenize(true_answer.lower())
            hypothesis = word_tokenize(pred_answer.lower())
            bleu = sentence_bleu([reference], hypothesis)
            bleu_scores.append(bleu)

        # 计算BERTScore（批量处理更高效）
        if len(bleu_scores) > 0:
            true_answers = []
            pred_answers = []
            for idx in indices[:len(bleu_scores)]:
                item = dataset[int(idx)]
                text_parts = item['text'].split("### Response: <|start_header_id|>assistant<|end_header_id|>")
                if len(text_parts) >= 2:
                    true_answers.append(text_parts[1].replace(EOS_TOKEN, "").strip())
                    # 重新生成以保持一致性
                    prompt_part = text_parts[0] + "### Response: <|start_header_id|>assistant<|end_header_id|>"
                    inputs = self.tokenizer([prompt_part], return_tensors="pt").to(self.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "### Response: <|start_header_id|>assistant<|end_header_id|>" in generated_text:
                        pred_answers.append(
                            generated_text.split("### Response: <|start_header_id|>assistant<|end_header_id|>")[
                                -1].strip())
                    else:
                        pred_answers.append(generated_text.strip())

            P, R, F1 = self.bert_scorer.score(pred_answers, true_answers)
            bert_scores['precision'] = P.cpu().numpy().tolist()
            bert_scores['recall'] = R.cpu().numpy().tolist()
            bert_scores['f1'] = F1.cpu().numpy().tolist()

        # 计算平均分数
        results = {
            'rouge1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            'rouge2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            'rougeL': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            'bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'bert_precision': np.mean(bert_scores['precision']) if bert_scores['precision'] else 0,
            'bert_recall': np.mean(bert_scores['recall']) if bert_scores['recall'] else 0,
            'bert_f1': np.mean(bert_scores['f1']) if bert_scores['f1'] else 0,
        }

        return results

    def evaluate_downstream_tasks(self, test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """评估下游任务性能"""
        self.model.eval()
        FastLanguageModel.for_inference(self.model)

        task_results = {
            'factual_accuracy': [],
            'relevance_score': [],
            'coherence_score': [],
            'completeness_score': []
        }

        for test_case in test_cases:
            question = test_case['question']
            context = test_case['context']
            expected_key_points = test_case.get('key_points', [])

            # 生成答案
            prompt = ft_prompt.format(question, context, "")
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response: <|start_header_id|>assistant<|end_header_id|>" in generated_text:
                answer = generated_text.split("### Response: <|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                answer = generated_text.strip()

            # 评估事实准确性（检查关键点是否被提及）
            if expected_key_points:
                mentioned_points = sum(1 for point in expected_key_points if point.lower() in answer.lower())
                factual_accuracy = mentioned_points / len(expected_key_points)
                task_results['factual_accuracy'].append(factual_accuracy)

            # 评估相关性（使用ROUGE-L作为代理）
            rouge_result = self.rouge_scorer.score(context, answer)
            task_results['relevance_score'].append(rouge_result['rougeL'].fmeasure)

            # 评估连贯性（基于句子长度和结构的简单度量）
            sentences = answer.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            coherence = 1.0 if 10 <= avg_sentence_length <= 25 else 0.5
            task_results['coherence_score'].append(coherence)

            # 评估完整性（答案长度作为代理）
            completeness = min(len(answer.split()) / 100, 1.0)  # 假设100词为完整答案
            task_results['completeness_score'].append(completeness)

        # 计算平均分数
        avg_results = {
            metric: np.mean(scores) if scores else 0
            for metric, scores in task_results.items()
        }

        return avg_results


# 创建自定义训练器，集成评估功能
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, evaluator: ModelEvaluator, swanlab_wrapper: SwanLabWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.swanlab_wrapper = swanlab_wrapper
        self.eval_history = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估方法，添加自定义指标"""
        # 调用父类的评估方法
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 获取实际的评估数据集
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # 添加自定义评估
        if eval_dataset is not None:
            try:
                # 计算困惑度
                perplexity = self.evaluator.calculate_perplexity(eval_dataset)
                # 将困惑度添加到metrics字典中
                metrics[f"{metric_key_prefix}_perplexity"] = perplexity

                # 计算生成质量指标
                gen_metrics = self.evaluator.evaluate_generation_quality(eval_dataset, num_samples=20)
                for key, value in gen_metrics.items():
                    metrics[f"{metric_key_prefix}_{key}"] = value

                # 记录到swanlab
                self.swanlab_wrapper.log({
                    f"{metric_key_prefix}/loss": metrics.get(f"{metric_key_prefix}_loss", 0),
                    f"{metric_key_prefix}/perplexity": perplexity,
                    f"{metric_key_prefix}/rouge1": gen_metrics['rouge1'],
                    f"{metric_key_prefix}/rouge2": gen_metrics['rouge2'],
                    f"{metric_key_prefix}/rougeL": gen_metrics['rougeL'],
                    f"{metric_key_prefix}/bleu": gen_metrics['bleu'],
                    f"{metric_key_prefix}/bert_f1": gen_metrics['bert_f1'],
                })

                # 保存评估历史
                self.eval_history.append({
                    'step': self.state.global_step,
                    'metrics': metrics
                })

                print(f"\n步骤 {self.state.global_step} 评估结果:")
                print(f"  Loss: {metrics.get(f'{metric_key_prefix}_loss', 'N/A'):.4f}")
                print(f"  Perplexity: {perplexity:.4f}")
                print(f"  ROUGE-1: {gen_metrics['rouge1']:.4f}")
                print(f"  BLEU: {gen_metrics['bleu']:.4f}")

            except Exception as e:
                print(f"评估时出现警告: {e}")

        return metrics


# 创建评估器实例
evaluator = ModelEvaluator(model, tokenizer)

# 修改训练参数，使用 eval_loss 作为最佳模型指标
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=1e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_perplexity",  # 修改为 eval_perplexity
    greater_is_better=False,  # 困惑度越小越好
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to=["none"],  # 避免与SwanLab冲突
    run_name="Financial_Analyst_with_Eval",
    ddp_timeout=1800,
    dataloader_pin_memory=False,
    disable_tqdm=False,
)

# 创建自定义训练器
trainer = CustomSFTTrainer(
    evaluator=evaluator,
    swanlab_wrapper=swanlab_wrapper,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)


# 主训练函数
def main_training():
    try:
        # 开始训练
        print("开始训练...")
        trainer_stats = trainer.train()

        # 训练后进行全面评估
        print("\n=== 最终模型评估 ===")

        # 1. 在验证集上的泛化性能
        val_perplexity = evaluator.calculate_perplexity(val_dataset)
        val_gen_metrics = evaluator.evaluate_generation_quality(val_dataset, num_samples=50)

        print(f"验证集困惑度: {val_perplexity:.4f}")
        print(f"验证集ROUGE-1: {val_gen_metrics['rouge1']:.4f}")
        print(f"验证集ROUGE-2: {val_gen_metrics['rouge2']:.4f}")
        print(f"验证集ROUGE-L: {val_gen_metrics['rougeL']:.4f}")
        print(f"验证集BLEU: {val_gen_metrics['bleu']:.4f}")
        print(f"验证集BERTScore F1: {val_gen_metrics['bert_f1']:.4f}")

        # 2. 下游任务评估
        downstream_test_cases = [
            {
                'question': "What are the main factors affecting the company's revenue growth?",
                'context': "The company experienced 15% revenue growth driven by expansion in Asian markets, new product launches, and strategic partnerships.",
                'key_points': ['Asian markets', 'new product', 'partnerships', '15%']
            },
            {
                'question': "Analyze the company's risk exposure in current market conditions.",
                'context': "Key risks include supply chain disruptions, currency fluctuations, regulatory changes, and intense competition in core markets.",
                'key_points': ['supply chain', 'currency', 'regulatory', 'competition']
            },
            {
                'question': "What is the outlook for the company's operating margins?",
                'context': "Operating margins are expected to improve by 200 basis points due to cost optimization initiatives and economies of scale.",
                'key_points': ['200 basis points', 'cost optimization', 'economies of scale']
            }
        ]

        downstream_results = evaluator.evaluate_downstream_tasks(downstream_test_cases)
        print(f"\n下游任务评估结果:")
        print(f"事实准确性: {downstream_results['factual_accuracy']:.4f}")
        print(f"相关性分数: {downstream_results['relevance_score']:.4f}")
        print(f"连贯性分数: {downstream_results['coherence_score']:.4f}")
        print(f"完整性分数: {downstream_results['completeness_score']:.4f}")

        # 记录最终评估结果到swanlab
        swanlab_wrapper.log({
            "final/val_perplexity": val_perplexity,
            "final/val_rouge1": val_gen_metrics['rouge1'],
            "final/val_rouge2": val_gen_metrics['rouge2'],
            "final/val_rougeL": val_gen_metrics['rougeL'],
            "final/val_bleu": val_gen_metrics['bleu'],
            "final/val_bert_f1": val_gen_metrics['bert_f1'],
            "final/downstream_factual_accuracy": downstream_results['factual_accuracy'],
            "final/downstream_relevance": downstream_results['relevance_score'],
            "final/downstream_coherence": downstream_results['coherence_score'],
            "final/downstream_completeness": downstream_results['completeness_score'],
        })

        # 保存评估报告
        evaluation_report = {
            'model_name': 'LLaMa3-Financial-Analyst',
            'training_steps': trainer.state.global_step,
            'validation_metrics': {
                'perplexity': val_perplexity,
                'rouge1': val_gen_metrics['rouge1'],
                'rouge2': val_gen_metrics['rouge2'],
                'rougeL': val_gen_metrics['rougeL'],
                'bleu': val_gen_metrics['bleu'],
                'bert_f1': val_gen_metrics['bert_f1'],
            },
            'downstream_metrics': downstream_results,
            'evaluation_history': trainer.eval_history,
            'timestamp': datetime.now().isoformat()
        }

        with open('evaluation_report.json', 'w') as f:
            json.dump(evaluation_report, f, indent=2)

        print("\n评估报告已保存至 evaluation_report.json")

        # 保存模型
        print("\n保存模型...")
        model.save_pretrained("/workspace/content/drive/LLaMa3-Financial-Analyst/Fine-l3_finagent_step60_eval")
        tokenizer.save_pretrained("/workspace/content/drive/LLaMa3-Financial-Analyst/l3_finagent_step60_eval")
        print("模型保存完成")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 确保清理资源
        print("\n执行最终清理...")

        # 清理trainer状态
        if hasattr(trainer, 'state'):
            try:
                trainer.state.save_to_json("trainer_state_final.json")
            except:
                pass

        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()

        # 等待所有操作完成
        time.sleep(2)


# 执行主训练函数
if __name__ == "__main__":
    try:
        main_training()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        # 最终清理将由atexit处理
        print("\n程序结束")
        sys.exit(0)
