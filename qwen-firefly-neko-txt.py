import re
import wave
import pyaudio
import time
import os
import soundfile as sf
import sys
import nltk
from tools.i18n.i18n import I18nAuto
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
sys.path.append('./GPT-SoVITS-v2-240821/GPT_SoVITS')
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
i18n = I18nAuto()
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=i18n(ref_language), 
                                   text=target_text, 
                                   text_language=i18n(target_language), top_p=1, temperature=1)
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

bat_file_path = 'GPT-SoVITS-v2-240821\\go-cli.bat'
model_name = "model/Qwen2.5-7B-Instruct"
print("初始化中...")
with open('background.txt', 'r', encoding='utf-8') as file:
        background = file.read()

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
print("初始化完成！输入exit退出")

while 1:
    prompt = input("用户：")
    if prompt == 'exit':
        break
    start_time = time.time()
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

    print("LLM 流萤猫酱：", end="", flush=True)
    response = ""
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True, buffer_size=1)
    model.generate(model_inputs.input_ids, streamer=streamer, max_new_tokens=512)
    for text in streamer:
        if text:
            print(text, end="", flush=True)
            response += text
    print("")
    print("流萤猫酱耗时：", time.time() - start_time)
    target_text = extract_language(response)
    
    with open('target_text.txt', 'w', encoding='utf-8') as file:
        file.write(target_text)
    synthesize("GPT_weights_v2/流萤-e10.ckpt", "SoVITS_weights_v2/流萤_e15_s810.pth", "firefly/ref_audio/example.wav", "ref_text.txt", "中文", "target_text.txt", "中文", "output")
    print("合成完成，耗时：", time.time() - start_time)
    #print("流萤猫酱:",response)
    play_wav("output/output.wav")
    