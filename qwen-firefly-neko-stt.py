import os
import time
import re
import wave
import pyaudio
import subprocess
import numpy as np
from funasr import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bat_file_path = 'GPT-SoVITS-v2-240821\\go-cli.bat'
model_name = "model/Qwen2.5-7B-Instruct"
print("初始化中...")

with open('background.txt', 'r', encoding='utf-8') as file:
        background = file.read()
with open('STT-background.txt', 'r', encoding='utf-8') as file:
        stt_background = file.read()

def extract_language(text):
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'【[^】]*】', '', text)
    return text
    
def play_wav(file_path):
    with wave.open(file_path, 'rb') as wf:
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()

# 使用 4 位量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
subprocess.run([bat_file_path], shell=True)

def correct(sentence):
    messages = [
        {"role": "system", "content": stt_background},
        {"role": "user", "content": sentence}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

model_dir = "model"

stt_model = AutoModel(
    model=f"{model_dir}/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
    vad_model=f"{model_dir}/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
    punc_model=f"{model_dir}/punc_ct-transformer_cn-en-common-vocab471067-large",  
    disable_update=True
)

def stt():
    chunk_size = 16000 * 3  # 3s
    #chunk_stride = chunk_size  # 确保每块长度足够

    # 初始化麦克风输入
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=chunk_size)

    cache = {}
    result_text = ""
    sound_threshold = 500
    wait_time = 1
    no_sound_start_time = time.time()
    try:
        while True:
            audio_data = stream.read(chunk_size)
            speech_chunk = np.frombuffer(audio_data, dtype=np.int16)
            if np.max(speech_chunk) > sound_threshold:
                # 保存音频块为临时文件
                temp_wav_path = "temp_chunk.wav"
                with wave.open(temp_wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(speech_chunk.tobytes())
                res = stt_model.generate(input=temp_wav_path, cache=cache, is_final=False, chunk_size=chunk_size)
                os.remove(temp_wav_path)
                #print(f"Model output: {res}")
                if res and len(res[0]["text"]) > 0:
                    result_text += res[0]["text"]
                    #corrected_text = correct(sentence=result_text)
                    print("未修改：", result_text)
                    #print("Qwen2.5修改：", corrected_text)
                    no_sound_start_time = time.time()
            else:
                if len(result_text) > 0 and time.time() - no_sound_start_time > wait_time:
                    print("已停顿，开始修正")
                    print("Qwen2.5修正：",correct(result_text))
                    no_sound_start_time = time.time()
                    return correct(result_text)
                
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

print("初始化完成！")

while True:
    prompt = stt()
    #if prompt == '退出':
    #    break
    messages = [
        {"role": "system", "content": background},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    target_text = extract_language(response)
    with open('GPT-SoVITS-v2-240821/target_text.txt', 'w', encoding='utf-8') as file:
        file.write(target_text)
    subprocess.run([bat_file_path], shell=True)
    print("流萤猫酱:",response)
    play_wav("GPT-SoVITS-v2-240821/output/output.wav")
    