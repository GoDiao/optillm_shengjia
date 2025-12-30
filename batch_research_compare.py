import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import httpx

# 导入原始的 DeepResearcher 类
from optillm.plugins.deep_research_plugin import DeepResearcher

# ============================================================================
# 全局请求限流器
# ============================================================================

class VLLMRequestThrottler:
    """确保全局同时只有 max_concurrent 个 vllm 请求"""
    def __init__(self, max_concurrent=8):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.Lock()
        self.active_requests = 0
        self.total_requests = 0
        self.waiting_requests = 0
        
    def __enter__(self):
        with self.lock:
            self.waiting_requests += 1
            waiting = self.waiting_requests
            active = self.active_requests
        
        # 如果需要等待，打印日志
        if waiting > 1:
            print(f"  ⏳ [Throttler] Request waiting... (active: {active}, waiting: {waiting})")
        
        self.semaphore.acquire()
        
        with self.lock:
            self.waiting_requests -= 1
            self.active_requests += 1
            self.total_requests += 1
            print(f"  ✅ [Throttler] Request started (active: {self.active_requests}, total: {self.total_requests})")
        
        return self
    
    def __exit__(self, *args):
        with self.lock:
            self.active_requests -= 1
            print(f"  ✅ [Throttler] Request finished (active: {self.active_requests})")
        
        self.semaphore.release()

# 全局限流器 - 设置为 vllm 的 max_num_seqs (默认 8)
_vllm_throttler = VLLMRequestThrottler(max_concurrent=6)


# ============================================================================
# 辅助函数
# ============================================================================

def ensure_dir(path):
    """Ensure directory exists"""
    if path:
        os.makedirs(path, exist_ok=True)


def read_jsonl(file_path):
    """Read JSONL file into list of dicts"""
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line: {line[:100]}... Error: {e}")
                continue
    return queries


def get_unprocessed_queries(queries_path, temp_result_dir):
    """Get queries that haven't been processed yet"""
    processed_query_ids = set()
    
    if os.path.exists(temp_result_dir):
        processed_query_filenames = os.listdir(temp_result_dir)
        processed_query_ids = {
            filename.split('.')[0] for filename in processed_query_filenames
            if filename.endswith('.json')
        }
    
    all_queries = read_jsonl(queries_path)
    unprocessed_queries = []
    
    for entry in all_queries:
        qid = str(entry.get("id", -1))
        if qid not in processed_query_ids:
            unprocessed_queries.append(entry)
    
    return unprocessed_queries


def get_query_logger(qid, log_dir):
    """Create a per-query logger that writes to its own file"""
    logger_name = f"Query_{qid}_{threading.current_thread().name}"
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"query_{qid}.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger


def write_temp_result(entry, temp_result_dir):
    """Write a single result to temp directory"""
    ensure_dir(temp_result_dir)
    qid = entry.get("id", -1)
    save_path = os.path.join(temp_result_dir, f"{qid}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=4)


def merge_temp_results(temp_result_dir, output_path):
    """Merge all temp results into final output file"""
    if not os.path.exists(temp_result_dir):
        print("No temp directory found")
        return
    
    metric_files = os.listdir(temp_result_dir)
    if not metric_files:
        print("No results to merge")
        return
    
    merged_metrics = []
    for metric_file in metric_files:
        if not metric_file.endswith('.json'):
            continue
        try:
            with open(os.path.join(temp_result_dir, metric_file), 'r', encoding='utf-8') as f:
                entry = json.load(f)
                merged_metrics.append(entry)
        except Exception as e:
            print(f"Error reading {metric_file}: {e}")
    
    merged_metrics.sort(key=lambda x: x.get('id', 0))
    
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else ".")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in merged_metrics:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Merged {len(merged_metrics)} results to {output_path}")


def save_run_statistics(stats_path, elapsed_time, num_workers, num_query, num_success, num_fail, 
                       start_time, end_time, run_label, model, base_url):
    """Save batch run statistics"""
    ensure_dir(os.path.dirname(stats_path) if os.path.dirname(stats_path) else ".")
    
    statistics = {
        "run_label": run_label,
        "model": model,
        "base_url": base_url,
        "num_workers": num_workers,
        "elapsed_time_seconds": round(elapsed_time, 2),
        "num_queries": num_query,
        "successful": num_success,
        "failed": num_fail,
        "success_rate": round(num_success / num_query * 100, 2) if num_query > 0 else 0,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat()
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Run statistics saved to {stats_path}")


# ============================================================================
# Thread-local storage for OpenAI clients
# ============================================================================

# 全局配置（会在运行时设置）
_thread_local = threading.local()
_global_config = {}


def set_global_config(base_url, api_key="dummy"):
    """设置全局配置"""
    global _global_config
    _global_config = {
        "base_url": base_url,
        "api_key": api_key
    }


def get_thread_local_client():
    """Get or create an OpenAI client for the current thread with timeout"""
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = OpenAI(
            base_url=_global_config.get("base_url"),
            api_key=_global_config.get("api_key", "dummy"),
            timeout=1200.0, 
            # 2. 核心：绝对禁止重试（防止雪崩）
            max_retries=0
        )
        print(f"  Created OpenAI client for thread {threading.current_thread().name}")
    return _thread_local.client


# ============================================================================
# LLMCallRecorder 类
# ============================================================================

class LLMCallRecorder:
    """记录和回放 LLM 调用的输入，用于性能对比测试"""
    
    def __init__(self, mode: str = "record", record_file: str = "llm_calls_record.json"):
        self.mode = mode
        self.record_file = record_file
        self.recorded_calls = []
        self.replay_index = 0
        self._lock = threading.Lock()  # 线程安全
        
        if mode == "replay":
            self._load_recorded_calls()
    
    def _load_recorded_calls(self):
        """从文件加载已记录的调用"""
        if os.path.exists(self.record_file):
            with open(self.record_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.recorded_calls = data.get("calls", [])
        else:
            raise FileNotFoundError(f"记录文件不存在: {self.record_file}")
    
    def save_recorded_calls(self, metadata: Dict[str, Any] = None):
        """保存记录的调用到文件"""
        with self._lock:
            data = {
                "metadata": metadata or {},
                "calls": self.recorded_calls,
                "total_calls": len(self.recorded_calls),
                "recorded_at": datetime.now().isoformat()
            }
            
            ensure_dir(os.path.dirname(self.record_file) if os.path.dirname(self.record_file) else ".")
            
            with open(self.record_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def record_call(self, call_name: str, system_prompt: str, user_prompt: str, 
                   temperature: float, max_tokens: int,
                   response_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录一次 LLM 调用的输入和输出"""
        with self._lock:
            call_record = {
                "call_index": len(self.recorded_calls),
                "call_name": call_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_metrics": response_data,
                "timestamp": datetime.now().isoformat()
            }
            self.recorded_calls.append(call_record)
            return call_record
    
    def get_next_call(self) -> Optional[Dict[str, Any]]:
        """获取下一个要回放的调用"""
        with self._lock:
            if self.replay_index >= len(self.recorded_calls):
                return None
            
            call = self.recorded_calls[self.replay_index]
            self.replay_index += 1
            return call


# ============================================================================
# DeepResearcherWithRecording 类 (继承 DeepResearcher)
# ============================================================================

class DeepResearcherWithRecording(DeepResearcher):
    """
    扩展 DeepResearcher 以支持输入录制和回放，并添加全局请求限流
    """
    
    def __init__(self, client, model: str, max_iterations: int = 5, 
                 max_sources: int = 30, logger=None,
                 recorder: Optional[LLMCallRecorder] = None,
                 run_label: str = "run1",
                 query_id: int = -1):
        super().__init__(client, model, max_iterations, max_sources, logger)
        
        self.recorder = recorder
        self.run_label = run_label
        self.query_id = query_id
    
    def _make_llm_call(self, system_prompt: str, user_prompt: str, 
                       temperature: float = 0.7, max_tokens: int = 1000, 
                       call_name: str = "LLM CALL") -> Any:
        """覆盖原方法，支持录制/回放模式，并添加全局限流"""
        # 确保输入是字符串
        if isinstance(system_prompt, (tuple, list)):
            system_prompt = system_prompt[0] if len(system_prompt) > 0 else ""
        if isinstance(user_prompt, (tuple, list)):
            user_prompt = user_prompt[0] if len(user_prompt) > 0 else ""
        
        system_prompt = str(system_prompt)
        user_prompt = str(user_prompt)
        
        # ========== 关键：使用限流器包裹整个 LLM 调用 ==========
        with _vllm_throttler:
            # === RECORD 模式 ===
            if self.recorder and self.recorder.mode == "record":
                start_time = datetime.now()
                
                if self.logger:
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"[{self.run_label}] LLM Call #{len(self.llm_calls) + 1}: {call_name}")
                    self.logger.info(f"Query ID: {self.query_id}, Thread: {threading.current_thread().name}")
                    self.logger.info(f"{'='*80}")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"❌ LLM call failed: {str(e)[:300]}")
                    raise
                
                end_time = datetime.now()
                latency_seconds = (end_time - start_time).total_seconds()
                prompt_details = getattr(response.usage, "prompt_tokens_details", None)
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
                # 记录响应数据
                response_data = {
                    "total_tokens": response.usage.total_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "latency_seconds": round(latency_seconds, 3),
                    "hit_tokens": cached_tokens,
                    "response_preview": response.choices[0].message.content.strip()[:500]
                }
                
                self.recorder.record_call(
                    call_name, system_prompt, user_prompt,
                    temperature, max_tokens, response_data
                )
                
                if self.logger:
                    response_text = response.choices[0].message.content.strip()
                    self.logger.info(f"Response: {response_text[:500]}...")
                    self.logger.info(f"Tokens: {response.usage.total_tokens}, Latency: {latency_seconds:.3f}s")
                
                # 更新指标
                self.total_tokens += response.usage.total_tokens
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                self.total_llm_latency_seconds += latency_seconds
                
                self.llm_calls.append({
                    "call_name": call_name,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "hit_tokens": cached_tokens,
                    "latency_seconds": round(latency_seconds, 3),
                    "timestamp": end_time.isoformat()
                })
                
                return response
            
            # === REPLAY 模式 ===
            elif self.recorder and self.recorder.mode == "replay":
                recorded_call = self.recorder.get_next_call()
                if recorded_call is None:
                    raise RuntimeError("回放调用已用尽")
                
                start_time = datetime.now()
                
                if self.logger:
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"[{self.run_label}] LLM Call #{len(self.llm_calls) + 1}: {call_name}")
                    self.logger.info(f"🔄 Replay (Index: {recorded_call['call_index']})")
                    self.logger.info(f"Query ID: {self.query_id}, Thread: {threading.current_thread().name}")
                    self.logger.info(f"Original Latency: {recorded_call['response_metrics']['latency_seconds']}s")
                    self.logger.info(f"{'='*80}")
                
                # 使用记录的输入
                messages = [
                    {"role": "system", "content": recorded_call["system_prompt"]},
                    {"role": "user", "content": recorded_call["user_prompt"]}
                ]
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=recorded_call["temperature"],
                        max_tokens=recorded_call["max_tokens"]
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"❌ LLM call failed: {str(e)[:300]}")
                    raise
                
                end_time = datetime.now()
                latency_seconds = (end_time - start_time).total_seconds()
                
                # 计算对比
                original_latency = recorded_call['response_metrics']['latency_seconds']
                latency_diff = latency_seconds - original_latency
                speedup = original_latency / latency_seconds if latency_seconds > 0 else 0
                
                if self.logger:
                    self.logger.info(f"Current Latency: {latency_seconds:.3f}s")
                    self.logger.info(f"📊 Diff: {latency_diff:+.3f}s, Speedup: {speedup:.2f}x")
                    self.logger.info(f"{'✅ LMCache faster' if latency_diff > 0 else '❌ LMCache slower'}")
                
                # 更新指标
                self.total_tokens += response.usage.total_tokens
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                self.total_llm_latency_seconds += latency_seconds
                
                self.llm_calls.append({
                    "call_name": call_name,
                    "call_index": recorded_call['call_index'],
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "hit_tokens": cached_tokens,
                    "latency_seconds": round(latency_seconds, 3),
                    "original_latency": original_latency,
                    "latency_diff": round(latency_diff, 3),
                    "speedup": round(speedup, 2),
                    "timestamp": end_time.isoformat()
                })
                
                return response
            
            # === 默认模式 ===
            else:
                return super()._make_llm_call(system_prompt, user_prompt, temperature, max_tokens, call_name)
    
    # ============================================================================
    # 覆盖 research 方法 - 根据模式选择执行策略
    # ============================================================================
    
    def research(self, system_prompt: str, initial_query: str):
        """
        覆盖 research 方法
        - record 模式: 正常执行（包括 web search）
        - replay 模式 + strict: 严格按顺序回放 LLM 调用
        - replay 模式 + skip_search: 执行逻辑但跳过搜索
        """
        
        # === REPLAY 模式 ===
        if self.recorder and self.recorder.mode == "replay":
            result, metrics = self._research_replay_mode_strict(system_prompt, initial_query)
            return result, metrics
            
        # === RECORD 模式：正常流程 ===
        result, metrics = super().research(system_prompt, initial_query)
        if self.recorder and self.recorder.mode == "record":
            metadata = {
                "run_label": self.run_label,
                "query_id": self.query_id,
                "model": self.model,
                "initial_query": initial_query,
                "max_iterations": self.max_iterations,
                "total_metrics": metrics.get("metrics", {})
            }
            self.recorder.save_recorded_calls(metadata)
            
        return result, metrics
    
    # ============================================================================
    # 严格回放模式
    # ============================================================================
    
    def _research_replay_mode_strict(self, system_prompt: str, initial_query: str):
        """
        严格回放模式：按照录制的顺序依次回放每个 LLM 调用
        不执行任何逻辑判断，只是机械地重放
        """
        start_time = datetime.now()
        
        self.logger.info("="*80)
        self.logger.info("🔄 [Strict Replay] Replaying recorded LLM calls in sequence")
        self.logger.info(f"   Total calls to replay: {len(self.recorder.recorded_calls)}")
        self.logger.info(f"   Query ID: {self.query_id}")
        self.logger.info(f"   Initial Query: {initial_query}")
        self.logger.info("="*80)
        
        try:
            last_response_content = ""
            
            # 依次回放每个 LLM 调用
            for idx, recorded_call in enumerate(self.recorder.recorded_calls):
                call_name = recorded_call.get('call_name', f'Call_{idx}')
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"🔄 [Replay {idx+1}/{len(self.recorder.recorded_calls)}] {call_name}")
                self.logger.info(f"{'='*80}")
                
                # 打印原始调用信息（用于调试）
                self.logger.info(f"📝 Original call info:")
                self.logger.info(f"   - Temperature: {recorded_call['temperature']}")
                self.logger.info(f"   - Max tokens: {recorded_call['max_tokens']}")
                self.logger.info(f"   - Original latency: {recorded_call['response_metrics']['latency_seconds']}s")
                self.logger.info(f"   - Original tokens: {recorded_call['response_metrics']['total_tokens']}")
                
                # 直接调用 _make_llm_call（它会自动处理 replay 逻辑）
                response = self._make_llm_call(
                    system_prompt=recorded_call['system_prompt'],
                    user_prompt=recorded_call['user_prompt'],
                    temperature=recorded_call['temperature'],
                    max_tokens=recorded_call['max_tokens'],
                    call_name=call_name
                )
                
                # 保存最后一次响应的内容
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    last_response_content = response.choices[0].message.content.strip()
                
                self.logger.info(f"✅ [Replay {idx+1}] Completed")
            
            # 使用最后一次调用的结果作为最终报告
            # 如果你想要更完整的报告，可以从录制的 metadata 中获取
            if last_response_content:
                final_report = last_response_content
            else:
                final_report = f"Strict replay completed for query {self.query_id}. See detailed metrics for performance comparison."
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("🔄 [Strict Replay] Summary")
            self.logger.info("="*80)
            self.logger.info(f"✅ Total calls replayed: {len(self.llm_calls)}")
            self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
            self.logger.info(f"🔢 Total tokens: {self.total_tokens}")
            self.logger.info(f"⚡ Total LLM latency: {self.total_llm_latency_seconds:.2f}s")
            self.logger.info(f"📊 Average latency per call: {self.total_llm_latency_seconds / len(self.llm_calls):.2f}s")
            self.logger.info("="*80)
            
            return (
                final_report, 
                {
                    "metrics": {
                        "total_time_seconds": round(total_time, 3),
                        "llm_calls": len(self.llm_calls),
                        "total_tokens": self.total_tokens,
                        "completion_tokens": self.completion_tokens,
                        "prompt_tokens": self.prompt_tokens,
                        "total_llm_latency_seconds": round(self.total_llm_latency_seconds, 3),
                        "avg_latency_per_call": round(self.total_llm_latency_seconds / len(self.llm_calls), 3) if self.llm_calls else 0,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "replay_mode": "strict",
                        "query_id": self.query_id,
                    },
                    "detailed_metrics": self.llm_calls
                }
            )
        
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error(f"❌ [Strict Replay] Failed at call {self.recorder.replay_index}/{len(self.recorder.recorded_calls)}")
            self.logger.error(f"   Error: {str(e)}")
            self.logger.error("="*80)
            self.logger.error("Full traceback:", exc_info=True)
            raise

# ============================================================================
# 处理单个查询（用于多线程）
# ============================================================================

def process_single_query(query_entry, model, system_prompt, request_config, 
                        log_dir, temp_result_dir, base_url, 
                        recorder_mode, records_dir, run_label):
    """
    处理单个查询（线程安全）
    """
    prompt = query_entry.get("prompt", "")
    qid = query_entry.get("id", -1)
    thread_name = threading.current_thread().name
    
    # 创建 per-query logger
    logger = get_query_logger(qid, log_dir)
    
    print(f"\n🔬 Query {qid}: Starting (thread: {thread_name})")
    logger.info("="*80)
    logger.info(f"Query {qid}: Starting deep research")
    logger.info(f"Run Label: {run_label}, Mode: {recorder_mode}")
    logger.info(f"Thread: {thread_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # 获取线程本地的 OpenAI client
        client = get_thread_local_client()
        
        # 创建 recorder
        record_file = f"{records_dir}/query_{qid}_record.json"
        
        if recorder_mode == "record":
            recorder = LLMCallRecorder(mode="record", record_file=record_file)
            logger.info(f"📝 Recording to {record_file}")
        elif recorder_mode == "replay":
            recorder = LLMCallRecorder(mode="replay", record_file=record_file)
            logger.info(f"🔄 Replaying from {record_file}")
        else:
            recorder = None
        
        # 创建 DeepResearcherWithRecording 实例
        researcher = DeepResearcherWithRecording(
            client=client,
            model=model,
            max_iterations=request_config.get("max_iterations", 5),
            max_sources=request_config.get("max_sources", 30),
            logger=logger,
            recorder=recorder,
            run_label=run_label,
            query_id=qid
        )
        
        # 执行研究
        result, metric_report = researcher.research(
            system_prompt=system_prompt,
            initial_query=prompt
        )
        
        elapsed = time.time() - start_time
        
        logger.info("="*80)
        logger.info(f"Query {qid} completed in {elapsed:.1f}s")
        logger.info("="*80)
        
        # 准备结果
        entry = {
            "id": qid,
            "prompt": prompt,
            "article": result,
            "endpoint": f"{base_url}/chat/completions",
            "model": model,
            "run_label": run_label,
            "recorder_mode": recorder_mode,
            "thread": thread_name,
            "elapsed_seconds": round(elapsed, 2),
            **(metric_report if metric_report is not None else {}),
        }
        
        write_temp_result(entry, temp_result_dir)
        print(f"✅ Query {qid}: Completed in {elapsed:.1f}s")
        
        # 清理 logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        return entry
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        logger.error("="*80)
        logger.error(f"Query {qid} failed after {elapsed:.1f}s")
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*80)
        
        print(f"❌ Query {qid}: Failed after {elapsed:.1f}s - {str(e)[:100]}")
        
        entry = {
            "id": qid,
            "prompt": prompt,
            "article": f"Error: {str(e)}",
            "endpoint": f"{base_url}/chat/completions",
            "model": model,
            "run_label": run_label,
            "thread": thread_name,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }
        write_temp_result(entry, temp_result_dir)
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        raise


# ============================================================================
# 对比两次运行结果
# ============================================================================

def compare_two_runs(lmcache_output_path, standard_output_path, comparison_output_path):
    """对比 LMCache 和标准 vllm 的运行结果"""
    print("\n" + "="*80)
    print("📊 生成性能对比报告")
    print("="*80 + "\n")
    
    # 加载结果
    lmcache_results = []
    with open(lmcache_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lmcache_results.append(json.loads(line))
    
    standard_results = []
    with open(standard_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                standard_results.append(json.loads(line))
    
    lmcache_results.sort(key=lambda x: x.get('id', 0))
    standard_results.sort(key=lambda x: x.get('id', 0))
    
    # 创建对比报告
    comparison = {
        "summary": {
            "lmcache": {
                "label": "LMCache vllm",
                "total_queries": len(lmcache_results),
                "total_time": sum(r.get("elapsed_seconds", 0) for r in lmcache_results),
                "avg_time_per_query": sum(r.get("elapsed_seconds", 0) for r in lmcache_results) / len(lmcache_results) if lmcache_results else 0,
                "total_llm_latency": sum(r.get("metrics", {}).get("total_llm_latency_seconds", 0) for r in lmcache_results),
            },
            "standard": {
                "label": "Standard vllm",
                "total_queries": len(standard_results),
                "total_time": sum(r.get("elapsed_seconds", 0) for r in standard_results),
                "avg_time_per_query": sum(r.get("elapsed_seconds", 0) for r in standard_results) / len(standard_results) if standard_results else 0,
                "total_llm_latency": sum(r.get("metrics", {}).get("total_llm_latency_seconds", 0) for r in standard_results),
            }
        },
        "per_query_comparison": []
    }
    
    for lm, std in zip(lmcache_results, standard_results):
        qid = lm.get("id")
        lmcache_time = lm.get("elapsed_seconds", 0)
        standard_time = std.get("elapsed_seconds", 0)
        time_saved = standard_time - lmcache_time
        speedup = standard_time / lmcache_time if lmcache_time > 0 else 0
        
        query_comp = {
            "query_id": qid,
            "prompt": lm.get("prompt", "")[:80] + "...",
            "lmcache_time": lmcache_time,
            "standard_time": standard_time,
            "time_saved": round(time_saved, 3),
            "speedup": round(speedup, 2),
            "lmcache_faster": time_saved > 0
        }
        
        if "metrics" in lm and "metrics" in std:
            lm_llm = lm["metrics"].get("total_llm_latency_seconds", 0)
            std_llm = std["metrics"].get("total_llm_latency_seconds", 0)
            query_comp["lmcache_llm_latency"] = lm_llm
            query_comp["standard_llm_latency"] = std_llm
            query_comp["llm_time_saved"] = round(std_llm - lm_llm, 3)
            query_comp["llm_speedup"] = round(std_llm / lm_llm, 2) if lm_llm > 0 else 0
        
        comparison["per_query_comparison"].append(query_comp)
    
    # 总体统计
    total_time_saved = comparison["summary"]["standard"]["total_time"] - comparison["summary"]["lmcache"]["total_time"]
    overall_speedup = comparison["summary"]["standard"]["total_time"] / comparison["summary"]["lmcache"]["total_time"] if comparison["summary"]["lmcache"]["total_time"] > 0 else 0
    
    llm_time_saved = comparison["summary"]["standard"]["total_llm_latency"] - comparison["summary"]["lmcache"]["total_llm_latency"]
    llm_speedup = comparison["summary"]["standard"]["total_llm_latency"] / comparison["summary"]["lmcache"]["total_llm_latency"] if comparison["summary"]["lmcache"]["total_llm_latency"] > 0 else 0
    
    comparison["summary"]["overall"] = {
        "total_time_saved": round(total_time_saved, 2),
        "overall_speedup": round(overall_speedup, 2),
        "lmcache_faster": total_time_saved > 0,
        "time_saved_percentage": round(total_time_saved / comparison["summary"]["standard"]["total_time"] * 100, 2) if comparison["summary"]["standard"]["total_time"] > 0 else 0,
        "llm_time_saved": round(llm_time_saved, 2),
        "llm_speedup": round(llm_speedup, 2),
    }
    
    # 保存
    ensure_dir(os.path.dirname(comparison_output_path) if os.path.dirname(comparison_output_path) else ".")
    with open(comparison_output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Comparison report saved to {comparison_output_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 性能对比摘要")
    print("="*80)
    print(f"\n🚀 LMCache vllm: {comparison['summary']['lmcache']['total_time']:.2f}s total, {comparison['summary']['lmcache']['avg_time_per_query']:.2f}s/query")
    print(f"🐢 Standard vllm: {comparison['summary']['standard']['total_time']:.2f}s total, {comparison['summary']['standard']['avg_time_per_query']:.2f}s/query")
    print(f"\n⚡ LMCache 加速: {total_time_saved:.2f}s saved ({comparison['summary']['overall']['time_saved_percentage']:.1f}%), {overall_speedup:.2f}x speedup")
    print(f"   LLM 延迟: {llm_time_saved:.2f}s saved, {llm_speedup:.2f}x speedup")
    print(f"   结论: {'✅ LMCache 更快' if total_time_saved > 0 else '❌ LMCache 更慢'}")
    print("="*80 + "\n")


# ============================================================================
# 批量运行函数（多线程）
# ============================================================================

def run_batch_research(
    queries_path: str,
    base_url: str,
    model: str,
    prefix: str,
    system_prompt: str,
    request_config: Dict,
    num_queries: int = None,
    max_workers: int = 8,
    recorder_mode: str = "record",
    record_dir: str = None,
    run_label: str = "lmcache"
):
    """
    批量运行研究任务（多线程并行，带全局请求限流）
    """
    # 目录设置
    temp_result_dir = f'{prefix}/temp_result'
    log_dir = f'{prefix}/logs'
    output_path = f"{prefix}/output.jsonl"
    stats_path = f'{prefix}/batch_run_stats.json'
    records_dir = f'{prefix}/records' if record_dir is None else record_dir
    
    # 创建目录
    ensure_dir(temp_result_dir)
    ensure_dir(log_dir)
    if recorder_mode == "record":
        ensure_dir(records_dir)
    
    # 设置全局配置（用于线程本地 client）
    set_global_config(base_url=base_url, api_key="dummy")
    
    # Setup main logger
    main_logger = logging.getLogger(f"BatchResearch_{run_label}")
    main_logger.setLevel(logging.INFO)
    
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    main_logger.addHandler(console_handler)
    
    main_logger.info("="*80)
    main_logger.info(f"Starting Batch Deep Research (Multi-threaded with Throttling): {run_label}")
    main_logger.info("="*80)
    
    # 读取查询
    queries = get_unprocessed_queries(queries_path, temp_result_dir)
    
    if num_queries is not None:
        queries = queries[:num_queries]
    # queries = [queries[0], queries[2]]
    total = len(queries)
    
    main_logger.info(f"Configuration:")
    main_logger.info(f"  - Queries file: {queries_path}")
    main_logger.info(f"  - Total queries: {total}")
    main_logger.info(f"  - Max workers: {max_workers}")
    main_logger.info(f"  - Request throttle limit: {_vllm_throttler.semaphore._value}")
    main_logger.info(f"  - Recorder mode: {recorder_mode}")
    if recorder_mode == "replay":
        main_logger.info(f"  - Records from: {records_dir}")
    main_logger.info(f"  - Model: {model}")
    main_logger.info(f"  - Base URL: {base_url}")
    main_logger.info("="*80)
    
    # 多线程处理
    successful = 0
    failed = 0
    start_time = datetime.now()
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Research") as executor:
        # 提交所有任务
        future_to_query = {
            executor.submit(
                process_single_query,
                query, model, system_prompt, request_config,
                log_dir, temp_result_dir, base_url,
                recorder_mode, records_dir, run_label
            ): query 
            for query in queries
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_query), 
                          total=len(future_to_query), 
                          desc="Processing queries"):
            query = future_to_query[future]
            qid = query.get("id", -1)
            
            try:
                future.result()
                successful += 1
            except Exception as e:
                failed += 1
                main_logger.error(f"Query {qid} failed: {str(e)[:100]}")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # 合并结果
    main_logger.info("="*80)
    main_logger.info("Processing Complete")
    main_logger.info(f"  - Total time: {elapsed:.1f}s")
    main_logger.info(f"  - Successful: {successful}/{total}")
    main_logger.info(f"  - Failed: {failed}/{total}")
    main_logger.info(f"  - Total vllm requests: {_vllm_throttler.total_requests}")
    main_logger.info("="*80)
    
    merge_temp_results(temp_result_dir, output_path)
    
    save_run_statistics(
        stats_path, elapsed, max_workers, total, successful, failed,
        start_time, end_time, run_label, model, base_url
    )
    
    main_logger.info(f"Results saved to: {output_path}")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Deep Research with LMCache Performance Testing (Multi-threaded with Request Throttling)")
    parser.add_argument("command", choices=["lmcache", "standard", "compare"], 
                       help="Command: lmcache (record), standard (replay), or compare")
    parser.add_argument("--queries", type=str, 
                       default="./deepresearchbench/query.jsonl",
                       help="Path to queries JSONL file")
    parser.add_argument("--num-queries", type=int, default=None,
                       help="Number of queries to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Max parallel workers (default: 4, recommended: 2-4)")
    parser.add_argument("--max-concurrent-requests", type=int, default=8,
                       help="Max concurrent vllm requests (default: 8, should match vllm max_num_seqs)")
    parser.add_argument("--lmcache-url", type=str,
                       default="http://10.170.27.246:8088/v1",
                       help="Base URL for LMCache vllm")
    parser.add_argument("--standard-url", type=str,
                       default="http://10.170.27.246:8089/v1",
                       help="Base URL for standard vllm")
    parser.add_argument("--model", type=str,
                       default="Qwen3-32B-128k",
                       help="Model name")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Max iterations per query")
    parser.add_argument("--max-sources", type=int, default=15,
                       help="Max sources per query")
    parser.add_argument("--lmcache-prefix", type=str, default="./reports/results_lmcache",
                       help="Output prefix for LMCache run")
    parser.add_argument("--standard-prefix", type=str, default="./reports/results_standard",
                       help="Output prefix for standard run")
    
    args = parser.parse_args()
    
    # 设置全局限流器
    global _vllm_throttler
    _vllm_throttler = VLLMRequestThrottler(max_concurrent=args.max_concurrent_requests)
    
    system_prompt = (
        "You are a research assistant. "
        "If the initial_query is in English, generate an English report; "
        "if it is in Chinese, generate the report in Chinese."
    )
    
    request_config = {
        "max_iterations": args.max_iterations,
        "max_sources": args.max_sources,
    }
    
    if args.command == "lmcache":
        print("\n" + "="*80)
        print(f"🚀 LMCache vllm (记录模式)")
        print(f"   Workers: {args.max_workers}, Request Limit: {args.max_concurrent_requests}")
        print("="*80 + "\n")
        
        run_batch_research(
            queries_path=args.queries,
            base_url=args.lmcache_url,
            model=args.model,
            prefix=args.lmcache_prefix,
            system_prompt=system_prompt,
            request_config=request_config,
            num_queries=args.num_queries,
            max_workers=args.max_workers,
            recorder_mode="record",
            run_label="lmcache_vllm"
        )
        
        print("\n✅ LMCache 运行完成！")
        print(f"📝 记录保存到: {args.lmcache_prefix}/records/")
        print("💡 下一步: python batch_research_mcompare_throttled.py standard\n")
        
    elif args.command == "standard":
        print("\n" + "="*80)
        print(f"🚀 标准 vllm (回放模式)")
        print(f"   Workers: {args.max_workers}, Request Limit: {args.max_concurrent_requests}")
        print("="*80 + "\n")
        
        lmcache_records_dir = f"{args.lmcache_prefix}/records"
        if not os.path.exists(lmcache_records_dir):
            print(f"❌ 错误: 找不到 LMCache 记录! ({lmcache_records_dir})")
            print("   请先运行: python batch_research_mcompare_throttled.py lmcache")
            return
        
        run_batch_research(
            queries_path=args.queries,
            base_url=args.standard_url,
            model=args.model,
            prefix=args.standard_prefix,
            system_prompt=system_prompt,
            request_config=request_config,
            num_queries=args.num_queries,
            max_workers=args.max_workers,
            recorder_mode="replay",
            record_dir=lmcache_records_dir,
            run_label="standard_vllm"
        )
        
        # 自动对比
        compare_two_runs(
            lmcache_output_path=f"{args.lmcache_prefix}/output.jsonl",
            standard_output_path=f"{args.standard_prefix}/output.jsonl",
            comparison_output_path="./comparison_report.json"
        )
        
    elif args.command == "compare":
        if not os.path.exists(f"{args.lmcache_prefix}/output.jsonl"):
            print(f"❌ 找不到 LMCache 结果!")
            return
        if not os.path.exists(f"{args.standard_prefix}/output.jsonl"):
            print(f"❌ 找不到标准 vllm 结果!")
            return
            
        compare_two_runs(
            lmcache_output_path=f"{args.lmcache_prefix}/output.jsonl",
            standard_output_path=f"{args.standard_prefix}/output.jsonl",
            comparison_output_path="./comparison_report.json"
        )


if __name__ == "__main__":
    '''
    python batch_research_mcompare_throttled.py lmcache \
        --queries ./deepresearchbench/query.jsonl \
        --num-queries 5 \
        --max-workers 1 \
        --max-concurrent-requests 1 \
        --lmcache-url http://10.170.27.246:8089/v1 \
        --model Qwen3-32B-128k-lmcache

    python batch_research_mcompare_throttled.py standard \
         --queries ./deepresearchbench/query.jsonl \
        --num-queries 3 \
        --max-workers 1 \
        --max-concurrent-requests 1 \
        --lmcache-url http://10.170.27.246:8089/v1 \
        --model Qwen3-32B-128k-lmcache

    '''
    main()
