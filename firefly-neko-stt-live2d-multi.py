import os
import time
import re
import wave
import pyaudio
import numpy as np
import concurrent.futures
import soundfile as sf
import sys
import nltk
from tools.i18n.i18n import I18nAuto
from funasr import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pygame
from pygame.locals import *
import live2d.v3 as live2d
from live2d.v3 import StandardParams
from live2d.utils import log
from live2d.utils.lipsync import WavHandler
sys.path.append('./GPT-SoVITS-v2-240821/GPT_SoVITS')
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
i18n = I18nAuto()
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

live2d.setLogEnable(False)

class QwenFireflyNeko:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        live2d.init()
        #self.audioPlayed = True
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
        self.running = True

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
            disable_update=True,
            ngpu=0 # 使用 CPU
        )

#live2d
    def live2d_init(self):
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL, vsync=1)
        pygame.display.set_caption("pygame window")

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.live2d_model = live2d.LAppModel()

        if live2d.LIVE2D_VERSION == 3:
            self.live2d_model.LoadModelJson(
                "Firefly-desktop/Firefly.model3.json"
            )

        self.live2d_model.Resize(*display)

        self.running = True

        # 关闭自动眨眼
        self.live2d_model.SetAutoBlinkEnable(False)
        # 关闭自动呼吸
        self.live2d_model.SetAutoBreathEnable(False)
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.scale: float = 1.0
        self.wavHandler = WavHandler()
        self.lipSyncN = 2.5
        fc = None
        sc = None
        self.live2d_model.StartRandomMotion("TapBody", 300, sc, fc)

        for i in range(self.live2d_model.GetParameterCount()):
            param = self.live2d_model.GetParameter(i)
            log.Debug(
                param.id, param.type, param.value, param.max, param.min, param.default
            )

        # 设置 part 透明度
        # log.Debug(f"Part Count: {model.GetPartCount()}")
        self.partIds = self.live2d_model.GetPartIds()
        self.currentTopClickedPartId = None

    def on_start_motion_callback(self, group: str, no: int):
        #log.Info("start motion: [%s_%d]" % (group, no))
        audioPath = "output/output.wav"
        pygame.mixer.music.load(audioPath)
        pygame.mixer.music.play()
        #self.audioPlayed = True
        #log.Info("start lipSync")
        self.wavHandler.Start(audioPath)

    def on_finish_motion_callback(self):
        #log.Info("motion finished")
        return

        # 获取全部可用参数
    
        # print(len(partIds))
        # log.Debug(f"Part Ids: {partIds}")
        # log.Debug(f"Part Id for index 2: {model.GetPartId(2)}")
        # model.SetPartOpacity(partIds.index("PartHairBack"), 0.5)

    

    def getHitFeedback(self, x, y):
        t = time.time()
        hitPartIds = self.live2d_model.HitPart(x, y, False)
        #print(f"hit part cost: {time.time() - t}s")
        #print(f"hit parts: {hitPartIds}")
        if self.currentTopClickedPartId is not None:
            pidx = self.partIds.index(self.currentTopClickedPartId)
            self.live2d_model.SetPartOpacity(pidx, 1)
            # model.SetPartScreenColor(pidx, 0.0, 0.0, 0.0, 1.0)
            self.live2d_model.SetPartMultiplyColor(pidx, 1.0, 1.0, 1., 1)
            # print("Part Screen Color:", model.GetPartScreenColor(pidx))
            #print("Part Multiply Color:", self.live2d_model.GetPartMultiplyColor(pidx))
        if len(hitPartIds) > 0:
            ret = hitPartIds[0]
            return ret

    
    def live2d_main(self):
        

        #audioPlayed = False
        self.live2d_model.SetExpression("expression2.exp3")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                    # currentTopClickedPartId = getHitFeedback(x, y)
                    # log.Info(f"Clicked Part: {currentTopClickedPartId}")
                    # model.Touch(x, y, onFinishMotionHandler=lambda : print("motion finished"), onStartMotionHandler=lambda group, no: print(f"started motion: {group} {no}"))
                    # model.StartRandomMotion(group="TapBody", onFinishMotionHandler=lambda : print("motion finished"), onStartMotionHandler=lambda group, no: print(f"started motion: {group} {no}"))
                    #model.SetRandomExpression()
                self.live2d_model.StartRandomMotion(priority=3)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.dx -= 0.1
                elif event.key == pygame.K_RIGHT:
                    self.dx += 0.1

                elif event.key == pygame.K_UP:
                    self.dy += 0.1

                elif event.key == pygame.K_DOWN:
                    self.dy -= 0.1

                elif event.key == pygame.K_i:
                    self.scale += 0.01

                elif event.key == pygame.K_u:
                    self.scale -= 0.01
                        
                elif event.key == pygame.K_r:
                    self.live2d_model.StopAllMotions()
                    self.live2d_model.ResetPose()
                        
                elif event.key == pygame.K_e:
                    self.live2d_model.ResetExpression()

            if event.type == pygame.MOUSEMOTION:
                        # 实现拖拽
                self.live2d_model.Drag(*pygame.mouse.get_pos())
                        # 测试性能？
                self.currentTopClickedPartId = self.getHitFeedback(*pygame.mouse.get_pos())
                        # pass

        self.live2d_model.Update()

        if self.currentTopClickedPartId is not None:
            pidx = self.partIds.index(self.currentTopClickedPartId)
            self.live2d_model.SetPartOpacity(pidx, 0.5)
                # 在此以 255 为最大灰度级
                # 原色和屏幕色取反并相乘，再取反
                # 以红色通道为例：r = 255 - (255 - 原色.r) * (255 - screenColor.r) / 255
                # 通道数值越大，该通道颜色对最终结果的贡献越大，下面的调用即为突出蓝色的效果
                # model.SetPartScreenColor(pidx, .0, 0., 1.0, 1)

                # r = multiplyColor.r * 原色.r / 255
                # 下面即为仅保留蓝色通道的结果
            self.live2d_model.SetPartMultiplyColor(pidx, .0, .0, 1., .9)

        if self.wavHandler.Update():
                # 利用 wav 响度更新 嘴部张合
            self.live2d_model.AddParameterValue(
                StandardParams.ParamMouthOpenY, self.wavHandler.GetRms() * self.lipSyncN
            )

        #if not self.audioPlayed:
                # 播放一个不存在的动作
        #    self.live2d_model.StartMotion(
        #        "",
        #        0,
        #        live2d.MotionPriority.FORCE,
        #        self.on_start_motion_callback,
        #        self.on_finish_motion_callback,
        #    )

            # 一般通过设置 param 去除水印
            # model.SetParameterValue("Param14", 1, 1)

        self.live2d_model.SetOffset(self.dx, self.dy)
        self.live2d_model.SetScale(self.scale)
        live2d.clearBuffer(1.0, 1.0, 1.0, 1)
        self.live2d_model.Draw()
        pygame.display.flip()
        pygame.time.wait(10)

#tts
    def synthesize(self, GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
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

    def extract_language(self, text):
        text = re.sub(r'（[^）]*）', '', text)
        text = re.sub(r'【[^】]*】', '', text)
        return text

    def play_wav(self, file_path):
        chunk_size = 1024
        with wave.open(file_path, 'rb') as wf:
            p = pyaudio.PyAudio()
            self.wavHandler.Start(file_path)
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
            

    def stt(self):
        p = pyaudio.PyAudio()
        chunk_size = 16000 * 3 # 3 秒
        stream = p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=chunk_size)
        try:
            while self.running:
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
                        self.no_sound_start_time = time.time()
                        return self.result_text
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

#llm
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
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.replace("流萤猫酱：", "")
        print("合成完成，耗时：", time.time() - start_time)
        print("已生成文本，正在合成语音...")
        target_text = self.extract_language(response)
        with open('target_text.txt', 'w', encoding='utf-8') as file:
            file.write(target_text)
        
        self.synthesize("GPT_weights_v2/流萤-e10.ckpt", 
                        "SoVITS_weights_v2/流萤_e15_s810.pth", 
                        "firefly/ref_audio/example.wav", 
                        "ref_text.txt", "中文", 
                        "target_text.txt", "中文", 
                        "output"
        )
        
        print("LLM 流萤猫酱:", response)
        self.play_wav("output/output.wav")

    def main(self):
        print("初始化完成！")
        self.live2d_init()
        with concurrent.futures.ThreadPoolExecutor() as executor: #ThreadPoolExecutor
            future_stt = executor.submit(self.stt)
            while self.running:
                if future_stt.done():
                    prompt = future_stt.result()
                    self.result_text = ""
                    executor.submit(self.process_llm, prompt)
                    future_stt = executor.submit(self.stt)

                self.live2d_main()

            live2d.dispose()
            pygame.quit()
            quit()

if __name__ == "__main__":
    app = QwenFireflyNeko()
    app.main()