# Qwen-Firefly-Neko

## 注意

该仓库仅存储了部分代码文件，GPT-SoVITS的核心文件以及所有大于100M的文件均被删除（包括模型文件和一些代码文件等等），肯定无法直接运行。具体运行方法看后文。

本仓库只能用于开源学习、娱乐目的，不可用于商业用途。

## 项目简介

`Qwen-Firefly-Neko` 是一个集成了live2d模型、语音转文字（STT）和大语言模型（LLM）处理的项目。该项目使用 PyTorch 和 Transformers 库，通过并行处理实现实时语音转文字和文本生成。正如名字所见，本项目构建了一个流萤猫酱的形象。

## 功能

- 实时语音转文字（STT）（使用`FunASR`）
- 文本生成和修正（LLM）（使用`Qwen2.5-7B-Instruct`模型）
- 并行处理 STT 和 LLM 任务
- 自动播放生成的语音文件（使用`GPT-SoVITS`，流萤语音模型使用的是b站白菜工厂1145号员工大佬的[GPT-SoVITS模型](https://www.bilibili.com/video/BV1sC411b7Ei/?spm_id_from=333.1387.upload.video_card.click&vd_source=76bb9f3f8ae762d5e5de82c84b34f583)，🤗[Hugging Face地址](https://huggingface.co/baicai1145/GPT-SoVITS-STAR)）
- 具有live2d模型（使用是b站依七哒大佬的[流萤前瞻小人模型](https://www.bilibili.com/video/BV1kJ4m1g7fs/?spm_id_from=333.1387.upload.video_card.click&vd_source=76bb9f3f8ae762d5e5de82c84b34f583))

## 使用方法

- 下载仓库内所有内容；

- 下载到本地后先将`GPT-SoVITS-v2-240821.zip`解压；

- 然后在该目录下输入`pip install -r requirements.txt`安装依赖，其中一些模块的下载需要依赖Microsoft Visual Studio，可下载vs然后选择“C/C++桌面开发”安装完后再用`pip`下载这些模块就不会`build wheels error`了；

- 接着直接运行`firefly-neko-stt-live2d-multi.py`即可。

## 注意事项
- 请确保在运行脚本前，所有依赖项和模型文件已正确安装和配置。
- 如果在运行过程中遇到任何问题，请检查依赖项版本和模型文件路径是否正确。
## 贡献
欢迎提交issues和pull requests来改进本项目。