import streamlit as st
import subprocess
import numpy as np
import soundfile as sf
import io
import os

st.title("Piper TTS Streamer")

# Input text box
default_text = """I carried everyone’s expectations on my back until my own voice disappeared beneath the weight.
I learned how to smile convincingly while something inside me slowly gave up."""
text_input = st.text_area("Enter text to speak:", value=default_text, height=150)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "en_US-hfc_female-medium.onnx")
SAMPLE_RATE = 22050

if st.button("Send"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
    else:
        status = st.empty()
        status.text("Initializing Piper...")

        try:
            # Start Piper process
            proc = subprocess.Popen(
                ["piper", "--model", MODEL_PATH, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Send text to Piper
            status.text("Generating audio...")
            proc.stdin.write(text_input.encode("utf-8"))
            proc.stdin.close()

            # Collect raw PCM audio
            raw_audio = proc.stdout.read()
            proc.wait()

            # Convert PCM → NumPy
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Write WAV to memory
            buffer = io.BytesIO()
            sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
            buffer.seek(0)

            status.success("Playback ready")
            st.audio(buffer, format="audio/wav")

        except Exception as e:
            status.error(f"Error occurred: {e}")
