# Box 自动化录音分析 Skill

这是一个可复用的录音分析自动化项目模板，用于快速搭建“自定义行业”的录音分析流程。
<img width="1149" height="799" alt="image" src="https://github.com/user-attachments/assets/c09602ea-9ee8-4e0a-ae4c-6d13c33316b5" />

#直接展示部分C端项目案例-销售录音分析（共计7大段，仅展示一部分)：

<img width="730" height="734" alt="image" src="https://github.com/user-attachments/assets/3eaa5b82-1246-4284-b134-65b6393104c4" />
<img width="741" height="716" alt="image" src="https://github.com/user-attachments/assets/ea7be764-0aba-4fc7-96a0-c4826b30e15c" />
<img width="734" height="621" alt="image" src="https://github.com/user-attachments/assets/39b6dd1f-8e0e-475d-9073-60fbc665b744" />


## 功能概览
- 音频切段：`ffmpeg`
- 语音转录：`SenseVoiceSmall`
- 语义检索：`ChromaDB + bge-large-zh-v1.5`
- 分析输出：结构化结果 + Markdown 报告
- 自动化编排：任务锁、失败回退、缓存清理

## 适配场景
- 销售/客服通话复盘
- 访谈内容提炼
- 会议录音总结
- 质检与评分场景

## 你需要做的定制
- 修改分析 Prompt（字段与评估规则）
- 填充 `knowledge_base/` 中的项目知识
- 调整报告结构
- 配置分析模型通道（OpenAI 兼容接口或 OpenClaw 会话）

## 快速开始

```bash
pip install -r requirements.txt
python vector_builder.py
python handler.py
```

详细执行规范见 `SKILL.md`，安装与排障见 `INSTALL.md`。
