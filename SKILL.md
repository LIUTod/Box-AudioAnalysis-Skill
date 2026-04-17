---
name: box-automation-audio-analysis
description: Execute Box automated recording-analysis workflow with deterministic dependency bootstrap, transcription, retrieval, and report generation. Use for customizable audio-analysis projects that require strict step-by-step execution.
---

# Box 自动化录音分析 Skill（执行规范版）

## 1) 适用范围
- 面向可自定义项目，不绑定具体行业
- 适用：通话复盘、访谈提炼、会议纪要、质检评分

## 2) 强约束（Agent 必须遵守）
1. 固定链路：`ffmpeg/ffprobe -> SenseVoiceSmall -> 向量检索 -> LLM 分析 -> 报告输出`
2. 固定顺序：任务锁 -> 依赖 -> 转录 -> 检索分析 -> 报告 -> 清理
3. 不得跳步：失败需走回退，不得直接进入后续步骤
4. 不得改道：未经用户确认，不得替换核心模型、工具和输出结构
5. 必须清理：成功/失败都要清理中间文件并释放锁

## 3) 组件职责
- `handler.py`：主流程编排与任务执行
- `dependency_manager.py`：系统识别与依赖准备
- `knowledge_rag.py`：向量编码、检索、引用拼接
- `vector_builder.py`：从 `knowledge_base/` 构建 `local_skills_db`

## 4) 标准流程（逐步执行）

### Step 0: 并发控制
- 检查 `.task_lock.json`
  - 无锁：继续
  - 有效锁：返回排队提示并结束
  - 超时锁：清理后继续
- 开始处理前加锁；结束时 `finally` 必须解锁

### Step 1: 输入与依赖
- 输入来源：
  - `audio_path`（附件）
  - 桌面目录 `原始录音/`
- 自动依赖准备：
  - 识别系统：`Windows/macOS/Linux`
  - 准备 `ffmpeg`/`ffprobe`
  - 准备 `SenseVoiceSmall`
  - 预热向量模型

### Step 2: 转录
- 用 `ffprobe` 获取时长
- 若时长 > 900s：
  - `ffmpeg` 按 600s 切段（16k 单声道 wav）
  - 逐段转录并合并
- 若切段失败或不可切段：回退整段转录

### Step 3: 检索与分析
- 从 `local_skills_db` 检索 Top-K（默认 3）参考片段
- 片段拼接为 `Reference Context`
- 构造结构化 Prompt，要求 JSON 输出
- 分析调用优先级：
  1) OpenAI 兼容接口（若已配置）
  2) `openclaw sessions send`
  3) 本地 fallback 分析

### Step 4: 报告输出
- 解析分析结果 JSON（失败回退 fallback）
- 渲染 Markdown 报告
- 保存到桌面 `录音分析/通用录音分析报告.md`
- 生成分段回复文本

### Step 5: 清理
- 清理 `data/`：
  - staged 音频
  - `*_transcript.txt`
  - `*_analysis_result.json`
  - `segment_tmp/`
- 释放 `.task_lock.json`

## 5) 回退策略
1. `ffmpeg/ffprobe` 不可用：整段转录 + 依赖缺失提示
2. 切段失败：回退整段转录
3. 向量检索失败：空引用上下文继续分析
4. 外部 LLM 失败：使用 `_fallback_analysis()`
5. 任意异常：返回错误，并继续执行清理与解锁

## 6) 禁止行为（防乱执行）
- 禁止绕过任务锁并发执行
- 禁止跳过清理步骤
- 禁止擅自替换核心链路
- 禁止把知识库全文直接塞给模型替代检索
- 禁止在用户未确认时执行无关写入

## 7) 知识库维护规范
- `knowledge_base/` 保持可填充框架
- 每个文件至少包含：目标、术语、规则、案例、边界
- 知识库更新后执行：

```bash
python vector_builder.py
```

## 8) Agent 输出规范
- 说明当前阶段（输入/转录/检索/分析/输出/清理）
- 失败时说明失败点与回退动作
- 报告生成后给出保存路径
- 最后明确“已清理缓存、已释放任务锁”

## 9) 快速运行
```bash
pip install -r requirements.txt
python handler.py
```
