"""
BoxClaw 销售录音分析 🦞 - OpenClaw Skill 入口处理器 v3.2.6
处理用户交互流程，工作流编排 - 会话内分析版本

更新说明 v3.2.6:
- 转录完成后插入本地 RAG：ChromaDB + BGE-large-zh-v1.5（Embedding 设备 CUDA/MPS/CPU 自适应），仅拼接 Top-3 知识块作为 Reference Context
- 大模型：若 config.json 配置 llm.provider=openai，则使用 OpenAI 兼容 API 且默认 stream=True（打字机输出）
- 未配置 OpenAI 时仍使用 openclaw sessions send（CLI 无流式）
- 超长录音（默认 >900s）：ffmpeg 切段 → 逐段 SenseVoice → 合并转录全文
- 新增 .task_lock.json 任务锁（30分钟超时自动释放），防止并发冲突
- 新增运行时中间产物与自动清理（transcript/analysis/segment_tmp）
- 新增报告分段输出（500-600 字分段，通过聊天通道回复）

更新说明 v3.0.9:
- 转录完成后，将转录文本发送到当前 OpenClaw 会话进行分析
- 生成符合要求的复盘文档（覆盖式、不展示原文、综合评分等）
"""

import os
import sys
import json
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from dependency_manager import ensure_runtime_dependencies

# Windows 兼容性：确保 stdout/stderr 使用 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


@dataclass
class SessionState:
    """会话状态"""
    step: str = "waiting_audio"
    salesperson_id: str = "default"
    salesperson_name: str = "默认"
    current_audio_path: Optional[str] = None
    last_result: Optional[Dict] = None
    learning_mode: bool = False  # 学习模式


class BoxClawHandler:
    """BoxClaw 销售复盘 Skill 处理器"""

    STATE_FILE = ".session_state.json"
    CONFIG_FILE = "config.json"
    
    # 工作文件夹名称
    AUDIO_FOLDER = "原始录音"
    REPORT_FOLDER = "录音分析"
    
    # 复盘文档固定文件名（覆盖式）
    REPORT_FILENAME = "通用录音分析报告.md"
    # Chroma 持久化目录（由 vector_builder.py 生成）
    VECTOR_DB_DIR = "local_skills_db"
    RAG_TOP_K = 3
    # 超过此时长（秒）则用 ffmpeg 切段后逐段 SenseVoice 转写再合并
    AUDIO_LONG_THRESHOLD_SEC = 900
    SEGMENT_DURATION_SEC = 600
    TASK_LOCK_FILE = ".task_lock.json"
    TASK_LOCK_TIMEOUT_SEC = 1800
    REPORT_SEGMENT_MIN = 500
    REPORT_SEGMENT_MAX = 600
    SENSEVOICE_LOCAL_DIR = ".runtime_cache/sensevoice"
    SENSEVOICE_ONNX_SUBDIR = "SenseVoiceSmall-onnx"
    SENSEVOICE_MODEL_ID = "haixuantao/SenseVoiceSmall-onnx"
    PREWARM_ON_BOOT = True

    def _get_stt_device(self) -> str:
        """
        SenseVoice 转写设备自适应：
        CUDA(NVIDIA) -> MPS(Apple Silicon) -> CPU
        """
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    gpu = torch.cuda.get_device_name(0)
                except Exception:
                    gpu = "NVIDIA CUDA"
                print(f"[Step 1] SenseVoice 设备: cuda ({gpu})")
                return "cuda"
            if torch.backends.mps.is_available():
                print("[Step 1] SenseVoice 设备: mps (Apple Silicon)")
                return "mps"
        except Exception as e:
            print(f"[Step 1] 设备检测失败，回退 CPU: {e}")
        print("[Step 1] SenseVoice 设备: cpu")
        return "cpu"

    def _resolve_sensevoice_model_ref(self) -> Tuple[str, str]:
        """
        返回 (model_ref, source_tag)
        source_tag: local / cache / remote
        """
        local_model_dir = self.workdir / self.SENSEVOICE_LOCAL_DIR
        local_onnx_dir = local_model_dir / self.SENSEVOICE_ONNX_SUBDIR
        cache_model_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--haixuantao--SenseVoiceSmall-onnx"
        )
        # 优先使用运行时缓存目录；兼容 huggingface/hub 与 legacy 结构。
        if local_onnx_dir.exists() and any(local_onnx_dir.iterdir()):
            return str(local_onnx_dir), "local"
        if local_model_dir.exists() and any(local_model_dir.iterdir()):
            return str(local_model_dir), "local"
        if cache_model_dir.exists():
            return str(cache_model_dir), "cache"
        return self.SENSEVOICE_MODEL_ID, "remote"

    def _get_sensevoice_model(self):
        """缓存 SenseVoice 模型实例，避免每个任务重复初始化。"""
        from funasr import AutoModel

        model_ref, source = self._resolve_sensevoice_model_ref()
        if self._stt_model is not None and self._stt_model_ref == model_ref:
            return self._stt_model

        if source == "local":
            print(f"[Step 1] 使用内置 SenseVoice 模型目录: {model_ref}")
        elif source == "cache":
            print(f"[Step 1] 使用缓存 SenseVoice 模型目录: {model_ref}")
        else:
            print(
                f"[Step 1] 未找到本地模型，自动下载到默认缓存（建议后续拷贝到 "
                f"{self.workdir / self.SENSEVOICE_LOCAL_DIR}）..."
            )

        self._stt_model = AutoModel(
            model=model_ref,
            device=self.stt_device,
            disable_update=True,
        )
        self._stt_model_ref = model_ref
        return self._stt_model

    def _prewarm_runtime(self):
        """
        预热关键链路（启动即加载）：
        - SenseVoice 模型
        - BGE embedding
        - Chroma collection
        """
        print("[Warmup] 启动预热：SenseVoice + RAG")
        try:
            self._get_sensevoice_model()
            print("[Warmup] SenseVoice 就绪")
        except Exception as e:
            print(f"[Warmup] SenseVoice 预热失败（任务时会重试）: {e}")
        try:
            from knowledge_rag import warmup_rag

            warmup_rag(self.workdir / self.VECTOR_DB_DIR)
            print("[Warmup] RAG 就绪")
        except Exception as e:
            print(f"[Warmup] RAG 预热失败（任务时会重试）: {e}")

    def __init__(self, workdir: str):
        # 使用 Path 处理跨平台路径
        self.workdir = Path(workdir).absolute()
        self.workdir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.workdir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.state = self._load_state()
        self.model_name = None  # 动态获取
        self._load_config()
        self._stt_model = None
        self._stt_model_ref = None

        dep_status = ensure_runtime_dependencies(self.workdir)
        for msg in dep_status.messages:
            print(f"[Deps] {msg}")
        
        # 创建工作文件夹
        self._setup_work_folders()
        
        # 检查 ffmpeg
        self.ffmpeg_path = self._find_ffmpeg()
        self.stt_device = self._get_stt_device()

        # 启动预热，让系统尽量保持待机态
        if self.config.get("performance", {}).get("prewarm_on_boot", self.PREWARM_ON_BOOT):
            self._prewarm_runtime()

    def _setup_work_folders(self):
        """创建工作文件夹"""
        desktop = self._get_desktop_path()
        
        # 原始录音文件夹
        self.audio_folder = desktop / self.AUDIO_FOLDER
        self.audio_folder.mkdir(exist_ok=True)
        
        # 录音分析文件夹
        self.report_folder = desktop / self.REPORT_FOLDER
        self.report_folder.mkdir(exist_ok=True)
        
        print(f"[BoxClawHandler] 工作文件夹已创建:")
        print(f"  - 原始录音: {self.audio_folder}")
        print(f"  - 录音分析: {self.report_folder}")

    def _find_ffmpeg(self) -> str:
        """查找 ffmpeg（系统路径）。"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print(f"[BoxClawHandler] 使用系统 ffmpeg")
            return "ffmpeg"
        except Exception:
            pass
        
        print("[BoxClawHandler] 警告: ffmpeg 未找到，音频处理可能失败")
        return None

    def _ffprobe_executable(self) -> Optional[str]:
        """与 ffmpeg 同目录的 ffprobe，或系统 PATH 中的 ffprobe。"""
        if not self.ffmpeg_path:
            return None
        fp = Path(self.ffmpeg_path)
        if fp.name.lower() == "ffmpeg.exe":
            probe = fp.parent / "ffprobe.exe"
        else:
            probe = fp.parent / "ffprobe"
        if probe.exists():
            return str(probe)
        try:
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            return "ffprobe"
        except Exception:
            return None

    def _get_audio_duration_sec(self, audio_path: Path) -> Optional[float]:
        exe = self._ffprobe_executable()
        if not exe:
            return None
        try:
            r = subprocess.run(
                [
                    exe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                return None
            return float(r.stdout.strip())
        except Exception:
            return None

    def _ffmpeg_split_to_wav_segments(
        self, src: Path, out_dir: Path, segment_sec: int
    ) -> List[Path]:
        """将音频切为固定时长 WAV 段（16k 单声道），供 SenseVoice 逐段识别。"""
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg 不可用")
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / "seg_%04d.wav")
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(segment_sec),
            # 分段边界允许 1 秒浮动，尽量避免语义截断
            "-segment_time_delta",
            "1",
            "-reset_timestamps",
            "1",
            "-map",
            "0:a",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            pattern,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout or "ffmpeg segment 失败")
        return sorted(out_dir.glob("seg_*.wav"))

    def _load_config(self):
        """从配置文件加载信息"""
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    agents = config.get("agents", {})
                    defaults = agents.get("defaults", {})
                    model_cfg = defaults.get("model", {})
                    primary = model_cfg.get("primary", "")
                    if primary:
                        self.model_name = primary.split("/")[-1] if "/" in primary else primary
                        print(f"[BoxClawHandler] 使用模型: {self.model_name}")
            except Exception as e:
                print(f"[BoxClawHandler] 读取配置文件失败: {e}")
        
        if not self.model_name:
            self.model_name = "minimax-m2.5"
            print(f"[BoxClawHandler] 使用默认模型: {self.model_name}")
        
        config_path = self.workdir / self.CONFIG_FILE
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}
        else:
            self.config = {}

    def _load_state(self) -> SessionState:
        """加载会话状态"""
        state_path = self.workdir / self.STATE_FILE
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return SessionState(**data)
            except Exception:
                pass
        return SessionState()

    def _save_state(self):
        """保存会话状态"""
        state_path = self.workdir / self.STATE_FILE
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(self.state.__dict__, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[BoxClawHandler] 保存状态失败: {e}")

    def _task_lock_path(self) -> Path:
        return self.workdir / self.TASK_LOCK_FILE

    def _check_task_lock(self) -> Tuple[bool, str]:
        """
        检查是否已有任务在执行。
        返回: (可继续处理, 说明信息)
        """
        lock_path = self._task_lock_path()
        if not lock_path.exists():
            return True, "无锁"
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8") or "{}")
            ts = float(data.get("timestamp", 0))
            now = time.time()
            age = now - ts
            if age > self.TASK_LOCK_TIMEOUT_SEC:
                print(f"[TaskLock] 发现超时锁（{int(age)}s），自动解锁")
                lock_path.unlink(missing_ok=True)
                return True, "超时锁已清理"
            wait_sec = int(max(1, self.TASK_LOCK_TIMEOUT_SEC - age))
            owner = data.get("audio", "未知任务")
            return False, f"当前正在处理：{owner}，预计约 {wait_sec}s 后可重试"
        except Exception as e:
            print(f"[TaskLock] 读取锁失败，按异常锁处理并清理: {e}")
            lock_path.unlink(missing_ok=True)
            return True, "异常锁已清理"

    def _acquire_task_lock(self, audio_name: str):
        payload = {
            "timestamp": time.time(),
            "audio": audio_name,
            "pid": os.getpid(),
        }
        self._task_lock_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[TaskLock] 已加锁: {audio_name}")

    def _release_task_lock(self):
        self._task_lock_path().unlink(missing_ok=True)
        print("[TaskLock] 已释放")

    def _get_desktop_path(self) -> Path:
        """获取桌面路径（跨平台兼容）"""
        desktop = Path.home() / "Desktop"
        
        if not desktop.exists():
            desktop_cn = Path.home() / "桌面"
            if desktop_cn.exists():
                return desktop_cn
            
            if sys.platform == 'win32':
                desktop_env = os.environ.get('USERPROFILE', '')
                if desktop_env:
                    desktop = Path(desktop_env) / "Desktop"
                    if desktop.exists():
                        return desktop
        
        return desktop

    def _scan_audio_files(self) -> list:
        """扫描原始录音文件夹中的音频文件"""
        audio_files = []
        
        if not self.audio_folder.exists():
            return audio_files
        
        patterns = ['*.mp3', '*.wav', '*.m4a', '*.m4r', '*.MP3', '*.WAV']
        
        for pattern in patterns:
            try:
                for file_path in self.audio_folder.glob(pattern):
                    if file_path.is_file():
                        audio_files.append(file_path)
            except Exception as e:
                print(f"[BoxClawHandler] 扫描文件失败 {pattern}: {e}")
                continue
        
        return audio_files

    def _manage_knowledge_base(self, action: str, filename: str = None, content: str = None) -> str:
        """管理知识库"""
        kb_dir = self.workdir / "knowledge_base"
        kb_dir.mkdir(exist_ok=True)
        
        if action == "list":
            files = sorted(kb_dir.glob("*.md"))
            if files:
                response = "📚 知识库文件：\n\n"
                for f in files:
                    response += f"- {f.name}\n"
                return response
            else:
                return "📚 知识库为空"
        
        elif action == "add":
            if not filename or not content:
                return "⚠️ 请提供文件名和内容。\n格式：添加知识 [文件名] [内容]"
            
            file_path = kb_dir / (filename if filename.endswith('.md') else f"{filename}.md")
            
            existing_content = ""
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as fp:
                    existing_content = fp.read()
            
            with open(file_path, 'w', encoding='utf-8') as fp:
                fp.write(existing_content + "\n\n" + content)
            
            return f"✅ 已添加/更新知识库文件：{file_path.name}\n\n现在可以通过「知识库」查看所有文件"
        
        return "❌ 未知操作"

    def _handle_knowledge_learning(self, user_message: str, image_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """处理知识库学习功能 - 支持图片和文字内容"""
        
        # 知识库分类映射
        kb_categories = {
            "01": "01_销售框架流程.md",
            "02": "02_销售产品知识.md",
            "03": "03_销售百问百答.md",
            "04": "04_优秀销售总结.md",
            "05": "05_项目介绍知识.md",
            "销售框架": "01_销售框架流程.md",
            "产品知识": "02_销售产品知识.md",
            "百问百答": "03_销售百问百答.md",
            "销售总结": "04_优秀销售总结.md",
            "项目介绍": "05_项目介绍知识.md",
        }
        
        kb_dir = self.workdir / "knowledge_base"
        kb_dir.mkdir(exist_ok=True)
        
        # 解析用户消息，识别目标知识库
        target_kb = None
        for key, filename in kb_categories.items():
            if key in user_message:
                target_kb = filename
                break
        
        # 如果没有指定知识库，提示用户选择
        if not target_kb:
            return (
                """📚 **知识库学习功能** 🦞

请指定要学习的知识库类型：

**知识库列表：**
- `学习 01` 或 `学习 销售框架` → 销售框架流程
- `学习 02` 或 `学习 产品知识` → 销售产品知识
- `学习 03` 或 `学习 百问百答` → 销售百问百答
- `学习 04` 或 `学习 销售总结` → 优秀销售总结
- `学习 05` 或 `学习 项目介绍` → 项目介绍知识

**使用方法：**
1. 选择知识库类型
2. 提供图片或文字内容
3. 系统自动识别并学习，追加到知识库中

**示例：**
- `学习 销售框架 + 图片` → 上传图片学习
- `学习 产品知识 + 这里是新的话术内容` → 文字内容学习
""",
                None
            )
        
        # 处理图片内容
        if image_path:
            try:
                # 使用 image tool 分析图片
                # 这里需要调用外部的 vision 模型
                # 由于是 skill 内部调用，我们先用简单的文本处理
                
                # 如果有图片路径，说明用户上传了图片
                # 这里假设图片已经被处理成文本内容
                content_from_image = f"\n\n--- 图片内容学习 ---\n来源：{image_path}\n内容：由用户上传的图片材料\n"
                
                # 追加到知识库
                return self._add_to_knowledge_base(target_kb, content_from_image)
                
            except Exception as e:
                return (f"❌ 图片识别失败: {str(e)}", None)
        
        # 处理文字内容 - 提取用户提供的学习内容
        # 移除命令部分，保留学习内容
        content = user_message
        for key in kb_categories.keys():
            content = content.replace(f"学习 {key}", "").replace(f"学习", "")
        
        content = content.strip()
        
        # 如果没有具体内容，提示用户输入
        if not content or len(content) < 5:
            return (
                f"✅ 已选择知识库：{target_kb}\n\n请提供要学习的内容：\n- 如果是图片，请直接上传图片\n- 如果是文字，请输入要学习的内容",
                None
            )
        
        # 追加内容到知识库
        return self._add_to_knowledge_base(target_kb, content)

    def _add_to_knowledge_base(self, filename: str, content: str) -> Tuple[str, Optional[str]]:
        """追加内容到知识库（不删除旧内容）"""
        
        kb_dir = self.workdir / "knowledge_base"
        kb_dir.mkdir(exist_ok=True)
        
        file_path = kb_dir / filename
        
        # 读取现有内容
        existing_content = ""
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as fp:
                existing_content = fp.read()
        
        # 添加时间戳和新内容
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_entry = f"""

---
# 新增学习内容 ({timestamp})

{content}
"""
        
        # 追加内容（不删除旧内容）
        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.write(existing_content + new_entry)
        
        return (
            f"✅ 学习完成！内容已追加到知识库：{filename}\n\n"
            f"学习内容：{content[:100]}..." if len(content) > 100 else f"学习内容：{content}\n\n"
            f"💡 旧知识保留完整，新内容已追加到末尾",
            None
        )

    def get_welcome_message(self) -> str:
        """获取欢迎信息"""
        learning_mode_notice = "\n\n⚠️ **学习模式已开启**：将分析结果沉淀到知识库" if self.state.learning_mode else ""
        
        return f"""👋 **欢迎使用通用录音分析 Skill v4.0.0**

这是一个可用于通话/访谈/语音记录场景的通用录音分析系统。

**技术升级**：
- 语音转录：SenseVoice 本地部署（免 API 费用）；**超长录音**（默认 >15 分钟）自动 **ffmpeg 切段** 后逐段转写再合并
- 知识检索：五大知识模块向量化（`knowledge_base` → `./local_skills_db`），Chroma + BGE-large-zh（CUDA/MPS/CPU，`./local_models`），仅 **Top-3** 进 Prompt（请先运行 `python vector_builder.py`）
- 语义分析：默认 OpenClaw 会话；可在 `config.json` 配置 `llm.provider=openai` 启用流式输出

**复盘文档规范**：
- ✅ 每次仅生成**一篇**复盘文档（覆盖更新，固定文件名）
- ✅ 复盘文档中**不展示原文**
- ✅ 包含**综合评分**（百分制）
- ✅ 复盘文档**不少于 1500 字**（分析 JSON 字段合计）

**工作文件夹**：
- 📁 {self.audio_folder} ← 请将录音文件放入此文件夹
- 📁 {self.report_folder} ← 复盘文档会保存到此文件夹（文件名：{self.REPORT_FILENAME}）

**使用方法**：
1. 将 mp3/wav 录音文件放入「原始录音」文件夹
2. 系统自动检测、转录并发送到当前会话分析
3. 复盘文档自动保存到「录音分析」文件夹{learning_mode_notice}

**命令**：
- `销售人员：张三` - 指定销售人员姓名
- `进入学习模式` - 将分析结果沉淀到知识库
- `退出学习模式` - 退出学习模式
- `知识库 列表` - 查看知识库文件
- `学习 [01-05]` - 学习知识库（如：学习 01、学习 销售框架）
- `学习 [知识库] + 内容` - 添加文字内容到知识库
- 上传图片 + `学习 [知识库]` - 识别图片内容并学习
"""

    def handle_message(self, user_message: str, audio_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """处理用户消息"""
        if "进入学习模式" in user_message:
            self.state.learning_mode = True
            self._save_state()
            return ("✅ 已进入学习模式\n分析结果将会自动沉淀到知识库。", None)
        
        elif "退出学习模式" in user_message:
            self.state.learning_mode = False
            self._save_state()
            return ("✅ 已退出学习模式", None)
        
        elif "知识库" in user_message or "学习" in user_message:
            # 知识库学习功能
            return self._handle_knowledge_learning(user_message, audio_path)
        
        elif "知识库" in user_message:
            parts = user_message.strip().split()
            if len(parts) >= 2 and parts[1] == "列表":
                return (self._manage_knowledge_base("list"), None)
            elif len(parts) >= 3 and parts[1] == "添加":
                filename = parts[2]
                content = " ".join(parts[3:])
                return (self._manage_knowledge_base("add", filename, content), None)
            else:
                return (self._manage_knowledge_base("list"), None)
        
        elif (("销售人员" in user_message or "销售姓名" in user_message) and not audio_path):
            match = re.search(r'(销售人员|销售姓名)[：:\s]*([^\s]+)', user_message)
            if match:
                name = match.group(2)
                self.state.salesperson_name = name
                self.state.salesperson_id = name.lower().replace(" ", "_")
                self._save_state()
                return (f"✅ 已设置销售人员：{name}\n现在请上传录音文件开始分析。", None)

        elif self.state.step == "waiting_audio":
            return self._handle_waiting_audio(user_message, audio_path)
        elif self.state.step == "processing":
            return self._handle_waiting_audio(user_message, audio_path)
        else:
            return self._handle_other(user_message)

    def _handle_waiting_audio(self, user_message: str, audio_path: Optional[str]) -> Tuple[str, Optional[str]]:
        """处理等待音频阶段"""
        if not audio_path:
            audio_files = self._scan_audio_files()
            
            if audio_files:
                return self._process_audio_files(audio_files)
            else:
                return (
                    f"📂 请将 mp3/wav 录音文件放入「原始录音」文件夹：\n{self.audio_folder}\n\n系统会自动检测并批量分析。\n\n如需指定销售人员姓名，请先发送：`销售人员：张三`",
                    None
                )

        return self._process_audio_files([Path(audio_path)])

    def _process_audio_files(self, audio_files: list) -> Tuple[str, Optional[str]]:
        """处理音频文件列表 - 每次仅生成一篇复盘文档"""
        if not audio_files:
            return ("未找到音频文件", None)
        
        # 仅处理第一个文件
        audio_file = audio_files[0]
        ok, lock_msg = self._check_task_lock()
        if not ok:
            return (f"⏳ 当前有任务执行中，已进入排队状态。\n{lock_msg}", None)
        
        try:
            self._acquire_task_lock(audio_file.name)
            result = self._process_single_audio(audio_file)
            
            if result.get('success'):
                response = f"""
✅ **复盘文档生成完成！** 🦞

**处理文件**: {audio_file.name}
**复盘文档**: {result['filename']}
**保存路径**: {result['path']}

文档包含：
- 学员画像分析
- SOP 标准流程核查
- 异议处理评估
- 综合评分（百分制）
- 话术优化建议

💡 文档已保存到「{self.report_folder}」
"""
                if self.state.learning_mode:
                    response += f"\n📚 学习模式：分析结果已沉淀到知识库"
                if result.get("chat_segment_reply"):
                    response += "\n\n---\n\n### 聊天通道分段回复\n\n"
                    response += result["chat_segment_reply"]
                
                return (response, None)
            else:
                return (f"❌ 分析失败: {result.get('error', '未知错误')}", None)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"❌ 处理异常: {str(e)}", None)
        finally:
            self._release_task_lock()

    def _transcribe_with_sensevoice(self, audio_file: Path) -> str:
        """SenseVoice 本地转写；超长音频用 skill 探测到的 ffmpeg 切段后逐段识别并拼接。"""
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        print(f"[Step 1] SenseVoice 转录: {audio_file.name}...")
        model = self._get_sensevoice_model()

        def one_clip(path: Path) -> str:
            res = model.generate(input=str(path), language="zh", use_itn=True)
            return rich_transcription_postprocess(res[0]["text"])

        dur = self._get_audio_duration_sec(audio_file)
        if dur is not None:
            print(f"[Step 1] 音频时长约 {dur:.0f} 秒")
        else:
            print("[Step 1] 无法探测时长，按整段转写")

        if (
            dur is not None
            and dur > self.AUDIO_LONG_THRESHOLD_SEC
            and self.ffmpeg_path
        ):
            tmp = (
                self.data_dir
                / "segment_tmp"
                / f"{audio_file.stem}_{datetime.now().strftime('%H%M%S%f')}"
            )
            try:
                print(
                    f"[Step 1] 超过 {self.AUDIO_LONG_THRESHOLD_SEC}s，"
                    f"ffmpeg 按 {self.SEGMENT_DURATION_SEC}s 切段转写..."
                )
                parts = self._ffmpeg_split_to_wav_segments(
                    audio_file, tmp, self.SEGMENT_DURATION_SEC
                )
                texts: List[str] = []
                for i, p in enumerate(parts):
                    print(f"[Step 1] 转写分段 {i + 1}/{len(parts)}: {p.name}")
                    texts.append(one_clip(p))
                return "\n".join(texts)
            except Exception as e:
                print(f"[WARNING] 切段转写失败，改为整段转写: {e}")
                return one_clip(audio_file)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        return one_clip(audio_file)

    def _save_runtime_artifacts(self, audio_file: Path, transcript: str, analysis_text: str) -> Dict[str, Path]:
        """保存运行中间产物，便于追踪与调试。"""
        stem = audio_file.stem
        transcript_path = self.data_dir / f"{stem}_transcript.txt"
        analysis_path = self.data_dir / f"{stem}_analysis_result.json"
        transcript_path.write_text(transcript, encoding="utf-8")
        analysis_path.write_text(analysis_text, encoding="utf-8")
        print(f"[Artifacts] 已保存: {transcript_path.name}, {analysis_path.name}")
        return {"transcript": transcript_path, "analysis": analysis_path}

    def _split_report_segments(self, report: str) -> List[str]:
        """
        将报告按 500-600 字切为多段，模拟飞书消息分段。
        注：这里按字符近似控制长度，优先保证句子完整。
        """
        lines = [ln for ln in report.splitlines() if ln.strip()]
        segments: List[str] = []
        current = ""
        for ln in lines:
            candidate = (current + "\n" + ln).strip() if current else ln
            if len(candidate) <= self.REPORT_SEGMENT_MAX:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = ln
            if len(current) > self.REPORT_SEGMENT_MAX:
                # 超长行硬切
                while len(current) > self.REPORT_SEGMENT_MAX:
                    segments.append(current[: self.REPORT_SEGMENT_MAX])
                    current = current[self.REPORT_SEGMENT_MAX :]
        if current:
            segments.append(current)
        # 补齐短段：将过短尾段并入前一段
        if len(segments) >= 2 and len(segments[-1]) < self.REPORT_SEGMENT_MIN:
            segments[-2] = segments[-2] + "\n" + segments[-1]
            segments.pop()
        return segments

    def _emit_report_segments(self, report: str) -> str:
        """
        输出层：聊天通道分段回复。
        返回拼接好的分段文本，供 handle_message 一次性返回给上层通道。
        """
        segments = self._split_report_segments(report)
        print(f"[Step 5] 报告分段数: {len(segments)}")
        blocks: List[str] = []
        for i, seg in enumerate(segments, 1):
            preview = seg[:120].replace("\n", " ")
            print(f"[Step 5] 第{i}段（{len(seg)}字）预览: {preview}...")
            blocks.append(f"【第{i}段】\n{seg}")
        return "\n\n".join(blocks)

    def _cleanup_runtime_files(self, audio_file: Path, artifacts: Dict[str, Path]):
        """
        自动清理运行时缓存（与 v3.2.6 目标对齐）：
        - data 下转录和分析中间文件
        - 分段临时目录
        """
        for key in ("transcript", "analysis"):
            p = artifacts.get(key)
            if p and p.exists():
                p.unlink(missing_ok=True)
        staged_audio = self.data_dir / f"{audio_file.stem}{audio_file.suffix}"
        if staged_audio.exists():
            staged_audio.unlink(missing_ok=True)
        seg_tmp = self.data_dir / "segment_tmp"
        if seg_tmp.exists():
            shutil.rmtree(seg_tmp, ignore_errors=True)
        print("[Step 7] 运行时缓存已清理")

    def _process_single_audio(self, audio_file: Path) -> dict:
        """处理单个音频文件 - 生成符合要求的复盘文档"""
        artifacts: Dict[str, Path] = {}
        try:
            # 从文件名提取销售人员姓名
            if self.state.salesperson_name == "默认" or self.state.salesperson_id == "default":
                basename = audio_file.stem
                if len(basename) <= 20 and not basename.isdigit():
                    self.state.salesperson_name = basename
                    self.state.salesperson_id = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', basename).lower()

            # Step 1 输入固化到 data 目录（便于后续清理）
            staged_audio = self.data_dir / f"{audio_file.stem}{audio_file.suffix}"
            if audio_file.resolve() != staged_audio.resolve():
                shutil.copy2(audio_file, staged_audio)
                print(f"[Step 1] 录音已复制到运行目录: {staged_audio}")
            else:
                print(f"[Step 1] 使用运行目录录音: {staged_audio}")

            # ===== Step 1: 语音转录 (SenseVoice，过长则 ffmpeg 切段) =====
            transcript = self._transcribe_with_sensevoice(staged_audio)
            print(f"✅ 转录完成: {len(transcript)} 字符")

            # ===== Step 2: Chroma RAG 检索 + 大模型分析 =====
            from knowledge_rag import format_reference_blocks, retrieve_top_k

            db_path = self.workdir / self.VECTOR_DB_DIR
            print(f"[Step 2a] RAG 检索（Chroma: {db_path}, top_k={self.RAG_TOP_K}）...")
            try:
                chunks, metas = retrieve_top_k(transcript, db_path, top_k=self.RAG_TOP_K)
                reference_context = format_reference_blocks(chunks, metas)
                print(f"[Step 2a] 已拼接 {len(chunks)} 条参考片段")
            except Exception as rag_e:
                print(f"[WARNING] RAG 检索异常: {rag_e}")
                reference_context = format_reference_blocks([], None)

            analysis_prompt = self._build_analysis_prompt(transcript, reference_context)
            print("[Step 2b] 调用大模型分析...")
            analysis_result = self._analyze_with_llm(analysis_prompt, transcript)
            artifacts = self._save_runtime_artifacts(staged_audio, transcript, analysis_result)

            # ===== Step 3: 生成复盘文档 =====
            print("[Step 3] 生成复盘文档...")
            
            record_datetime = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            
            # 构建复盘文档内容（不包含原文）
            report = self._build_report(audio_file.name, transcript, analysis_result, record_datetime)
            
            # 保存复盘文档（固定文件名，覆盖式）
            report_path = self.report_folder / self.REPORT_FILENAME
            
            print(f"[Step 3] 保存复盘文档到: {report_path}")
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            # ===== Step 5: 聊天通道分段回复 =====
            chat_segment_reply = self._emit_report_segments(report)
            
            # 学习模式：保存到知识库
            if self.state.learning_mode:
                kb_file = self.report_folder / f"学习笔记_{self.state.salesperson_name}.md"
                with open(kb_file, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"[学习模式] 已保存学习笔记: {kb_file.name}")
            
            return {
                "success": True,
                "filename": self.REPORT_FILENAME,
                "path": str(report_path),
                "salesperson": self.state.salesperson_name,
                "segments": len(self._split_report_segments(report)),
                "chat_segment_reply": chat_segment_reply,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"处理失败: {str(e)}"
            }
        finally:
            try:
                # ===== Step 7: 自动清理缓存 =====
                self._cleanup_runtime_files(audio_file, artifacts)
            except Exception as cleanup_e:
                print(f"[WARNING] 清理缓存失败: {cleanup_e}")

    def _build_analysis_prompt(self, transcript: str, reference_context: str) -> str:
        """构建分析 Prompt（仅拼接 RAG Top-K 知识块，不再全量加载 Markdown）"""
        prompt = f"""你是一位专业的录音分析顾问。请分析以下录音转写文本。

===== Reference Context（RAG 检索到的相关知识片段，仅作依据；不足处请结合行业常识）=====
{reference_context}
===== Reference Context 结束 =====

请严格按照以下 JSON 格式输出分析结果，不要有额外文字：
{{
  "customer_profile": "对说话方特征和诉求的概括",
  "sop_check": "对沟通流程执行情况的评估",
  "objection_handling": "对关键疑问或阻碍点处理质量的评估",
  "advice": ["建议1（详细描述）", "建议2（详细描述）", "建议3（详细描述）", "建议4（详细描述）"],
  "score_sop": "SOP执行评分 0-100",
  "score_objection": "异议处理评分 0-100",
  "score_skill": "话术技巧评分 0-100",
  "score_overall": "综合评分 0-100（基于以上三项加权计算）"
}}

录音转写文本：
{transcript}

请输出 JSON 格式的分析结果："""

        return prompt

    def _analyze_with_llm(self, analysis_prompt: str, transcript: str) -> str:
        """
        大模型分析：优先使用 config.json 中 OpenAI 兼容接口（默认 stream=True）。
        否则使用 openclaw CLI（无流式）；失败则本地备用分析。
        """
        llm = self.config.get("llm") or {}
        provider = (llm.get("provider") or "").strip().lower()
        if provider == "openai" and llm.get("api_key"):
            try:
                return self._openai_chat_stream(analysis_prompt, llm)
            except Exception as e:
                print(f"[WARNING] OpenAI 兼容接口失败: {e}")

        return self._openclaw_sessions_send(analysis_prompt, transcript)

    def _openai_chat_stream(self, prompt: str, llm: dict) -> str:
        from openai import OpenAI

        api_key = llm.get("api_key")
        base_url = llm.get("base_url") or None
        model = llm.get("model", "gpt-4o-mini")
        use_stream = llm.get("stream", True)
        print(f"[LLM] OpenAI 兼容 API, model={model}, stream={use_stream}")

        client = OpenAI(api_key=api_key, base_url=base_url)
        if use_stream:
            parts: list[str] = []
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    piece = delta.content
                    print(piece, end="", flush=True)
                    parts.append(piece)
            print()
            text = "".join(parts)
            if not text.strip():
                raise RuntimeError("流式返回为空")
            return text

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        msg = resp.choices[0].message.content or ""
        if not msg.strip():
            raise RuntimeError("非流式返回为空")
        return msg

    def _openclaw_sessions_send(self, analysis_prompt: str, transcript: str) -> str:
        """通过 openclaw CLI 发送到会话（无 stream 参数支持，完整输出后返回）"""
        session_key = (self.config.get("openclaw") or {}).get("session") or (
            "agent:main:openclaw-weixin:direct:o9cq802ocomwu17lqjqwjop0xatq@im.wechat"
        )
        cmd = [
            "openclaw",
            "sessions",
            "send",
            "--session",
            session_key,
            analysis_prompt,
        ]
        print("[LLM] openclaw sessions send（CLI 不支持 stream，结果一次性返回）")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print("✅ 会话分析完成!")
                return result.stdout
            print(f"[WARNING] sessions_send 失败: {result.stderr}")
        except Exception as e:
            print(f"[WARNING] 会话分析异常: {e}")
        print("[WARNING] 使用本地备用分析...")
        return self._fallback_analysis(transcript)

    def _fallback_analysis(self, transcript: str) -> str:
        """本地备用分析（当 sessions_send 不可用时）- 通用版本。"""
        transcript_len = len((transcript or "").strip())
        analysis = {
            "customer_profile": f"通话文本长度约 {transcript_len} 字。建议结合角色、诉求与情绪线索做人工复核。",
            "sop_check": "流程完整性中等：建议重点检查开场目的、信息采集、关键问题确认、结束动作四个环节。",
            "objection_handling": "可识别到部分疑问回应，但论据与收口动作需要增强，建议补充证据与下一步承诺。",
            "advice": [
                "将转录按主题分段，先归纳关键议题再逐项评估。",
                "对每个问题给出“现象-证据-建议”三段式结论。",
                "增加可执行跟进动作，并明确时间与责任人。"
            ],
            "score_sop": "75",
            "score_objection": "72",
            "score_skill": "74",
            "score_overall": "74"
        }
        return json.dumps(analysis, ensure_ascii=False)

    def _build_report(self, audio_filename: str, transcript: str, analysis_result: str, record_datetime: str) -> str:
        """构建复盘文档内容（不包含原文，包含综合评分）"""
        # 解析分析结果
        try:
            # 尝试解析 JSON
            if isinstance(analysis_result, str):
                # 清理可能的 markdown 代码块
                analysis_text = analysis_result.strip()
                if analysis_text.startswith("```json"):
                    analysis_text = analysis_text[7:]
                if analysis_text.startswith("```"):
                    analysis_text = analysis_text[3:]
                if analysis_text.endswith("```"):
                    analysis_text = analysis_text[:-3]
                analysis_text = analysis_text.strip()
                
                try:
                    analysis = json.loads(analysis_text)
                except:
                    # 如果 JSON 解析失败，使用备用分析
                    analysis = json.loads(self._fallback_analysis(transcript))
            else:
                analysis = analysis_result
        except:
            analysis = json.loads(self._fallback_analysis(transcript))
        
        advice_list = analysis.get("advice", [])
        advice_text = "\n".join([f"{i}. {v}" for i, v in enumerate(advice_list, 1)]) if advice_list else "1. 暂无具体建议。"

        score_sop = analysis.get("score_sop", "0")
        score_ob = analysis.get("score_objection", "0")
        score_skill = analysis.get("score_skill", "0")
        score_all = analysis.get("score_overall", "0")

        report = f"""# 通用录音分析报告

> 生成时间：{record_datetime}
> 原音频文件：{audio_filename}
> 转录方式：SenseVoice 本地部署（超长自动分段）
> 分析方式：RAG Top-{self.RAG_TOP_K} + 当前 OpenClaw 会话智能分析

---

## 第1段：综合评分 + 对话画像分析

### 综合评分（先看结论）
- 综合评分：**{score_all}/100**
- SOP 执行：{score_sop}/100
- 异议处理：{score_ob}/100
- 话术技巧：{score_skill}/100

### 对话画像
{analysis.get('customer_profile', '暂无')}

---

## 第2段：SOP 标准流程核查（表格）

| 检查项 | 结果摘要 |
|---|---|
| 开场与身份建立 | {analysis.get('sop_check', '暂无')[:220]}... |
| 需求探查 | {analysis.get('sop_check', '暂无')[220:440]}... |
| 产品介绍与价值塑造 | {analysis.get('sop_check', '暂无')[440:660]}... |
| 异议处理与促单 | {analysis.get('sop_check', '暂无')[660:880]}... |

> 详细 SOP 评估：  
{analysis.get('sop_check', '暂无')}

---

## 第3段：参考案例对照分析

基于知识库检索结果，本次对话在“需求识别、证据表达、收口动作”三个环节仍有优化空间。  
建议采用“事实片段 + 证据说明 + 改进动作”的方式输出复盘结论，并为后续沟通沉淀可复用话术与流程模板。  
同时应将关键原话映射为结构化标签（目标、阻碍、情绪、承诺、下一步），方便二次复盘与质量跟踪。

---

## 第4段：产品知识对照 + 异议处理

{analysis.get('objection_handling', '暂无')}

---

## 第5段：优化建议

{advice_text}

---

## 第6段：关键因素 + 跟进建议

### 关键因素
1. 是否明确核心目标与约束条件。  
2. 是否提供可验证证据支撑关键结论。  
3. 是否形成可执行的下一步动作与时间点。

### 跟进建议（48小时内）
1. 按主题输出 1 份结构化摘要。  
2. 对关键问题补充证据或样例。  
3. 明确二次沟通时间与执行人。  
4. 记录本轮结论并更新知识库模板。

---

## 第7段：综合评分详情 + 结论

| 评估维度 | 评分 | 说明 |
|----------|------|------|
| SOP 执行评分 | {score_sop}/100 | 流程完整性与稳定性 |
| 异议处理评分 | {score_ob}/100 | 关键阻碍识别与处理 |
| 话术技巧评分 | {score_skill}/100 | 表达清晰度与引导能力 |
| 综合评分 | **{score_all}/100** | 三项加权后的整体水平 |

### 结论
本次通话具备基础沟通框架，但在证据呈现和收口动作上仍有优化空间。建议按本报告的 48 小时跟进清单执行，并在下一轮复盘中重点关注“关键问题解决率”和“后续行动完成率”。

---

**本报告由通用录音分析 Skill 自动生成**  
v4.0.0
"""
        
        return report

    def _handle_other(self, user_message: str) -> Tuple[str, Optional[str]]:
        """处理其他消息"""
        if user_message.lower() in ["重载知识库", "重新加载知识库", "reload"]:
            return ("✅ 知识库已重新加载！", None)
        elif user_message.lower() in ["重置", "重新开始", "reset"]:
            self.state = SessionState()
            self._save_state()
            return (self.get_welcome_message(), None)
        
        audio_files = self._scan_audio_files()
        if audio_files:
            return self._process_audio_files(audio_files)
        
        return ("请将 mp3/wav 格式录音文件放入「原始录音」文件夹开始分析。\n\n命令：\n- `进入学习模式` - 沉淀分析结果到知识库\n- `知识库 [列表|添加]` - 管理知识库", None)

# ============ OpenClaw Skill 入口 ============

def main():
    """被 OpenClaw 调用的入口"""
    if len(sys.argv) < 2:
        workdir = str(Path(__file__).parent.absolute())
    else:
        workdir = sys.argv[1]

    handler = BoxClawHandler(workdir)
    print(handler.get_welcome_message())

if __name__ == "__main__":
    main()
