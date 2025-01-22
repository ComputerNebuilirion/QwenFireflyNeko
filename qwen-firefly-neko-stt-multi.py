import os
import time
import re
import wave
import pyaudio
import subprocess
import numpy as np
import concurrent.futures
import soundfile as sf
import sys
import nltk
from tools.i18n.i18n import I18nAuto
from funasr import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
sys.path.append('./GPT-SoVITS-v2-240821/GPT_SoVITS')
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
i18n = I18nAuto()
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

class QwenFireflyNeko:
    def __init__(self):
        self.bat_file_path = 'GPT-SoVITS-v2-240821\\go-cli.bat'
        self.model_name = "model/Qwen2.5-7B-Instruct"
        print("初始化中...")

        with open('background.txt', 'r', encoding='utf-8') as file:
            self.background = file.read()
        with open('STT-background.txt', 'r', encoding='utf-8') as file:
            self.stt_background = file.read()

        self.end_of_talk = False
        self.cache = {}
        self.result_text = ""
        self.sound_threshold = 500
        self.wait_time = 1
        self.no_sound_start_time = time.time()

        # 使用 4 位量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_dir = "model"
        self.stt_model = AutoModel(
            model=f"{model_dir}/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
            vad_model=f"{model_dir}/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
            punc_model=f"{model_dir}/punc_ct-transformer_cn-en-common-vocab471067-large",  
            disable_update=True
        )

    def synthesize(self, GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
        # Read reference text
        print("hi")
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

    def extract_language(self, text):
        text = re.sub(r'（[^）]*）', '', text)
        text = re.sub(r'【[^】]*】', '', text)
        return text

    def play_wav(self, file_path):
        chunk_size = 1024
        with wave.open(file_path, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            stream.stop_stream()
            stream.close()
            p.terminate()

    def correct(self, text):
        messages = [
            {"role": "system", "content": self.stt_background},
            {"role": "user", "content": text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def stt(self):
        p = pyaudio.PyAudio()
        chunk_size = 16000 * 3 # 3 秒
        stream = p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=chunk_size)
        try:
            while True:
                audio_data = stream.read(chunk_size)
                speech_chunk = np.frombuffer(audio_data, dtype=np.int16)
                if np.max(speech_chunk) > self.sound_threshold:
                    # 保存音频块为临时文件
                    self.end_of_talk = False
                    temp_wav_path = "temp_chunk.wav"
                    with wave.open(temp_wav_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(16000)
                        wf.writeframes(speech_chunk.tobytes())
                    res = self.stt_model.generate(input=temp_wav_path, cache=self.cache, is_final=False, chunk_size=chunk_size)
                    os.remove(temp_wav_path)
                    if res and len(res[0]["text"]) > 0:
                        self.result_text += res[0]["text"]
                        print("STT 未修改：", self.result_text)
                        self.no_sound_start_time = time.time()
                else:
                    if not self.end_of_talk and len(self.result_text) > 0 and time.time() - self.no_sound_start_time > self.wait_time:
                        print("已停顿")
                        self.end_of_talk = True
                        #corrected_text = self.correct(self.result_text)
                        #print("STT Qwen2.5修正：", corrected_text)
                        self.no_sound_start_time = time.time()
                        return self.result_text
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def process_llm(self, prompt):
        start_time = time.time()
        messages = [
            {"role": "system", "content": self.background},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        #print("LLM 流萤猫酱：", end="", flush=True)
        #response = ""
        #streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True, buffer_size=1)
        #self.model.generate(model_inputs.input_ids, streamer=streamer, max_new_tokens=512)
        #for text in streamer:
        #    if text:
        #        print(text, end="", flush=True)
        #        response += text
        #print("")

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("合成完成，耗时：", time.time() - start_time)
        print("已生成文本，正在合成语音...")
        target_text = self.extract_language(response)
        with open('target_text.txt', 'w', encoding='utf-8') as file:
            file.write(target_text)
        
        self.synthesize("GPT_weights_v2/流萤-e10.ckpt", "SoVITS_weights_v2/流萤_e15_s810.pth", "firefly/ref_audio/example.wav", "ref_text.txt", "中文", "target_text.txt", "中文", "output")
        print("合成完成，耗时：", time.time() - start_time)
        print("LLM 流萤猫酱:", response)
        self.play_wav("output/output.wav")

    def main(self):
        print("初始化完成！")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                start_time = time.time()
                prompt = self.stt()
                print("合成完成，耗时：", time.time() - start_time)
                self.result_text = ""
                executor.submit(self.process_llm, prompt)

if __name__ == "__main__":
    app = QwenFireflyNeko()
    app.main()