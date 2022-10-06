import copy
import json
import multiprocessing as mp
import numpy as np
import soundcard as sc
import sounddevice as sd
import vosk
import speech_recognition as sr
from datetime import datetime as dt

r = sr.Recognizer()

now = dt.now()
time = now.strftime("%d/%m/%y %H:%M:%S")

def capture_audio_output(audio_queue: mp.Queue,
                         capture_sec: float,
                         sample_rate: int) -> None:
    
    num_frame: int = int(sample_rate * capture_sec)
    
    while True:
        audio = sc.get_microphone(include_loopback=True, id=str(sc.default_speaker().name)) \
            .record(numframes=num_frame, samplerate=sample_rate, blocksize=sample_rate)
        audio_queue.put(copy.copy(audio[:, 0]))


def speech_to_text(audio_queue: mp.Queue,
                   sample_rate: int) -> None:
    NO_LOG: int = -1
    MODEL_PATH = "model-en-md"
    FILE = "speech.json"
    
    vosk.SetLogLevel(NO_LOG)
    
    model: vosk.Model = vosk.Model(model_path=MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    
    print("Recognizer is ready")
    print("Output sound from a speaker or a headphone")
    print("#" * 40)
    
    while True:
        audio = audio_queue.get()
        audio = map(lambda x: (x+1)/2, audio)
        audio = np.fromiter(audio, np.float16)
        audio = audio.tobytes()

        if recognizer.AcceptWaveform(audio):
            result: json = json.loads(recognizer.Result())
            text = result["text"]
            
            if text != "":
                # write text to json file
                try:
                    # get the old data
                    with open(FILE, mode='r', encoding='utf-8') as data:
                        file_data = json.load(data)

                    # append the new data
                    with open(FILE, mode='w', encoding='utf-8') as new_data:
                        new_text = {
                            'datetime': time,
                            'text': text
                        }
                        file_data.append(new_text)
                        json.dump(file_data, new_data)

                except FileNotFoundError:
                    print("File not found...")


def main():
    CAPTURE_SEC: float = 0.4
    
    audio_queue: mp.Queue = mp.Queue()
    sample_rate: int = int(sd.query_devices(kind="output")["default_samplerate"])
    stt_proc: mp.Process = mp.Process(target=speech_to_text,
                                      args=(audio_queue, sample_rate))
    
    print("Type Ctrl+C to stop")
    
    stt_proc.start()
    
    try:
        capture_audio_output(audio_queue=audio_queue, capture_sec=CAPTURE_SEC, sample_rate=sample_rate)
        stt_proc.join()
    except KeyboardInterrupt:
        stt_proc.terminate()
        
        print("\nDone")


if __name__ == "__main__":
    main()
