## Optillm Deep Research Research Engine 流程解析
research 函数是整个 Deep Research Engine（深度研究引擎） 的大脑和指挥官。它串联起了所有的组件（搜索、阅读、写作、评估），是一个典型的 Agentic Workflow（智能体工作流）
### 第一阶段：启动与初始草稿 (Initialization & Warm Start)
这一阶段的目标是：不要从零开始瞎编，先找点资料，搭个靠谱的架子。

1 分解问题 (decompose_query)

  * 动作：拿到用户的大问题（例如“2025年AI芯片市场分析”），让 LLM 把它拆解成 3-5 个具体的搜索子问题（如“2025 AI芯片市场规模”、“英伟达 vs 华为昇腾 市场份额”、“AI芯片技术趋势”）。

  * 目的：大问题很难直接搜索，拆细了搜得更准。

2 执行初始搜索 (perform_web_search)

  * 动作：拿着上面的子问题去搜索引擎跑一遍。

  * 结果：得到一堆初始的网页摘要（Snippets）。

3 生成初步草稿 (Draft Generation)

  * 分叉逻辑（关键点）：

    * 路径 A (Warm Start - 推荐)：如果搜索成功了，调用 generate_preliminary_draft_with_initial_search。LLM 会结合刚才搜到的信息写草稿，并打上初始的引用标签 [1], [2]。

    * 路径 B (Cold Start - 兜底)：如果网断了或者没搜到东西，调用 generate_preliminary_draft。LLM 只能凭记忆写，并在不知道的地方打上 [NEEDS RESEARCH] 标签。

  * 产物：self.current_draft —— 这就是我们的“初始状态”，它可能比较粗糙，有很多占位符，但结构已经有了。

### 第二阶段：扩散去噪循环 (The Iterative Denoising Loop)
这是算法的核心，对应代码中的 for iteration in range(self.max_iterations):。 这个过程就像 **“降噪”**：把“未知的占位符（噪声）”替换成“确凿的事实（信号）”。

1 差距分析 (analyze_draft_gaps)

  * 动作：LLM 变身为“审稿人”，阅读当前的草稿。

  * 任务：找出所有的 [NEEDS RESEARCH]、[SOURCE NEEDED] 以及逻辑薄弱的地方。

  * 产出：一个 gaps 列表，每个 gap 包含具体的搜索关键词和优先级。

2 针对性搜索 (perform_gap_targeted_search)

  * 动作：只针对上面找到的缺口进行搜索。

  * 逻辑：优先处理 HIGH 优先级的缺口。这比第一阶段的广撒网要精准得多。

3 内容提取 (extract_and_fetch_urls)

  * 动作：从搜索结果中提取信息，并更新 self.citations（引用字典）。

4 去噪/整合 (denoise_draft_with_retrieval)

  * 动作：LLM 变身为“编辑”。

  * 输入：旧草稿 + 新搜到的填坑信息。

  * 任务：把新信息填入旧草稿的对应位置，替换掉占位符，并加上引用。

  * 结果：生成了新的 current_draft。

5 质量评估 (evaluate_draft_quality)

  * 动作：LLM 再次变身为“打分员”。

  * 任务：对比新旧草稿，给完整性、准确性、深度打分（0.0 - 1.0）。

6 早停检查 (Termination Check)

  * 逻辑：如果 completeness > 0.9（已经很完美了）或者 improvement < 0.03（这轮折腾半天也没啥长进），就提前跳出循环（break）。

  * 目的：省钱、省时间，防止过拟合。

### 第三阶段：收尾与出版 (Finalization)
循环结束（要么是跑满了 max_iterations，要么是质量达标提前退出了），进入最后一步。

1 最终抛光 (finalize_research_report)

  * 动作：

  * 格式化：确保标题漂亮，段落清晰。

  * 清洗：暴力移除所有剩下的 [NEEDS RESEARCH] 标签（不能让读者看到这些）。

  * 检查：确保所有引用格式正确。

  * 验证：代码内部还会跑一个 validate_report_completeness，如果发现问题尝试自动修复。

2 生成参考文献列表

  * 动作：这是纯 Python 逻辑，不是 LLM 生成的。

  * 逻辑：扫描正文中到底用了哪些 [x]，然后从数据库里把对应的 URL 和标题拿出来，拼在文章最后。

3 返回结果

输出最终的 Markdown 报告和详细的 Token 消耗/耗时统计。
