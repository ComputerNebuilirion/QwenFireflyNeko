# Qwen-Firefly-Neko

## 注意

该仓库仅存储了部分代码文件，GPT-SoVITS的核心文件以及所有大于100M的文件均被删除（包括模型文件和一些代码文件等等），肯定无法直接运行，可通过链接直接从网盘下载一键包。

*链接暂未提供，因为文件夹太大了:(


## 项目简介

`Qwen-Firefly-Neko` 是一个集成了live2d模型、语音转文字（STT）和大语言模型（LLM）处理的项目。该项目使用 PyTorch 和 Transformers 库，通过并行处理实现实时语音转文字和文本生成。正如名字所见，本项目构建了一个流萤猫酱的形象。

## 功能

- 实时语音转文字（STT）（使用`FunASR`）
- 文本生成和修正（LLM）（使用`Qwen2.5-7B-Instruct`模型）
- 并行处理 STT 和 LLM 任务
- 自动播放生成的语音文件（使用`GPT-SoVITS`）
- 具有live2d模型（使用是b站依七哒大佬的[流萤前瞻小人模型](https://www.bilibili.com/video/BV1kJ4m1g7fs/?spm_id_from=333.1387.upload.video_card.click&vd_source=76bb9f3f8ae762d5e5de82c84b34f583))

## 依赖

请确保安装以下依赖项：(`pip install`)

- PyTorch
- Transformers
- FunASR
- PyAudio
- NumPy
- jieba_fast
- g2p_en
- wordsegment
- live2d-py

其中一些模块的下载需要依赖Microsoft Visual Studio，可下载vs然后选择“C/C++桌面开发”安装完后再用`pip`下载这些模块就不会`build wheels error`了。

## 使用方法

建议从网盘链接下载一键包。

*链接暂未提供，因为文件夹太大了:(

## 代码说明
### 初始化
在 QwenFireflyNeko 类的 `__init__` 方法中，初始化了音频流、模型和相关配置。

### 语音转文字（STT）
`stt` 方法负责从音频中提取文本，并在检测到停顿时调用 `correct` 方法进行文本修正。(`correct`方法未实装，因为这样太慢了)

### 文本生成和修正（LLM）
`process_llm` 方法负责处理 LLM 的生成任务。

### live2d模型渲染
`live2d_main`方法负责live2d模型的渲染和对口型。

### 并行处理
在 `main` 方法中，使用 `concurrent.futures.ThreadPoolExecutor` 来并行执行`live2d_main`、 `stt` 和 `process_llm`。每次从 `stt` 获取到新的 `prompt` 后，都会提交一个新的任务给线程池来处理 LLM 的生成任务。

## 注意事项
- 请确保在运行脚本前，所有依赖项和模型文件已正确安装和配置。
- 如果在运行过程中遇到任何问题，请检查依赖项版本和模型文件路径是否正确。
## 贡献
欢迎提交issues和pull requests来改进本项目。