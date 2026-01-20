import subprocess
import sounddevice as sd
import numpy as np

text = '''I carried everyone’s expectations on my back until my own voice disappeared beneath the weight.
I learned how to smile convincingly while something inside me slowly gave up.
Nights grew louder with thoughts I never dared to speak out loud. Even now, when the world feels distant and cold, I keep moving, not because I am strong, but because stopping feels worse. I don’t know when it gets better, but I’m still here, breathing, waiting.'''
model = "en_US-hfc_female-medium.onnx"
sr = 22050

proc = subprocess.Popen(
    ["piper", "--model", model, "--output-raw"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

proc.stdin.write(text.encode())
proc.stdin.close()

with sd.OutputStream(samplerate=sr, channels=1, dtype="int16") as stream:
    while True:
        chunk = proc.stdout.read(2048)
        if not chunk:
            break
        audio = np.frombuffer(chunk, dtype=np.int16)
        stream.write(audio)

proc.wait()
