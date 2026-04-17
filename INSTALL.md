# Box 自动化录音分析 Skill - 安装与启动

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行时自动准备依赖

启动后程序会尝试自动准备：
- `ffmpeg/ffprobe`
- `SenseVoiceSmall` 模型缓存
- 向量模型缓存

如自动准备失败，请按日志提示手动安装。

## 3. 构建向量库（首次或知识库更新后）

```bash
python vector_builder.py
```

## 4. 启动

```bash
python handler.py
```

## 5. 验证
- 能识别音频并转录
- 能检索知识库片段
- 能生成报告并落盘
- 任务完成后能释放 `.task_lock.json` 且清理 `data/` 中间文件
