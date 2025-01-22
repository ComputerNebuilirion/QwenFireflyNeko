# Qwen-Firefly-Neko-STT-Multi

## 注意

该仓库仅存储了部分代码文件，所有大于100M的文件均被删除（包括模型文件和一些代码文件等等），肯定无法直接运行，可通过下述链接直接从网盘下载一键包。

## 项目简介

`Qwen-Firefly-Neko-STT` 是一个集成了语音转文字（STT）和大语言模型（LLM）处理的项目。该项目使用 PyTorch 和 Transformers 库，通过并行处理实现实时语音转文字和文本生成。正如名字所见，本项目构建了一个流萤猫酱的形象。

## 功能

- 实时语音转文字（STT）（使用`FunASR`）
- 文本生成和修正（LLM）（使用`Qwen2.5-7B-Instruct`模型）
- 并行处理 STT 和 LLM 任务
- 自动播放生成的语音文件（使用`GPT-SoVITS`）

## 依赖

请确保安装以下依赖项：

- Python 3.8+
- PyTorch
- Transformers
- FunASR
- PyAudio
- NumPy
- jieba_fast
- g2p_en
- wordsegment

可以使用以下命令安装依赖项：

```bash
pip install torch transformers funasr pyaudio numpy
```

## 使用方法
1.克隆项目到本地：
```
git clone https://github.com/yourusername/Qwen-Firefly-Neko-STT-Multi.git
cd Qwen-Firefly-Neko-STT-Multi
```
2.确保模型文件夹和配置文件已正确放置在对应目录下。WIP

3.运行主脚本：
```
python qwen-firefly-neko-stt-multi.py
```

## 代码说明
### 初始化
在 QwenFireflyNekoSTT 类的 `__init__` 方法中，初始化了音频流、模型和相关配置。

### 语音转文字（STT）
`stt` 方法负责从音频中提取文本，并在检测到停顿时调用 `correct` 方法进行文本修正。

### 文本生成和修正（LLM）
`process_llm` 方法负责处理 LLM 的生成任务。

### 并行处理
在 `main` 方法中，使用 `concurrent.futures.ThreadPoolExecutor` 来并行执行 `stt` 和 `process_llm`。每次从 `stt` 获取到新的 `prompt` 后，都会提交一个新的任务给线程池来处理 LLM 的生成任务。

## 注意事项
- 请确保在运行脚本前，所有依赖项和模型文件已正确安装和配置。
- 如果在运行过程中遇到任何问题，请检查依赖项版本和模型文件路径是否正确。
## 贡献
欢迎提交issues和pull requests来改进本项目。