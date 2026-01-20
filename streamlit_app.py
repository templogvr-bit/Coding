import streamlit as st
import subprocess
import sounddevice as sd
import numpy as np
import os

st.title("Piper TTS Streamer")

# Input text box
default_text = """I carried everyone’s expectations on my back until my own voice disappeared beneath the weight.
I learned how to smile convincingly while something inside me slowly gave up."""
text_input = st.text_area("Enter text to speak:", value=default_text, height=150)

# Model configuration
model_path = "en_US-hfc_female-medium.onnx"
sample_rate = 22050

if st.button("Send"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
    else:
        status_placeholder = st.empty()
        status_placeholder.text("Initializing Piper...")

        try:
            # Start the piper process
            proc = subprocess.Popen(
                ["piper", "--model", model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            
            # Send text to piper
            status_placeholder.text("Generating and streaming audio...")
            proc.stdin.write(text_input.encode())
            proc.stdin.close()
            
            # Stream audio to speakers
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
                while True:
                    chunk = proc.stdout.read(2048) # Read in chunks
                    if not chunk:
                        break
                    audio = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio)
            
            proc.wait()
            status_placeholder.success("Done!")
            
        except Exception as e:
            status_placeholder.error(f"Error occurred: {e}")
