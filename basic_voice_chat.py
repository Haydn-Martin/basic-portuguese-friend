import io
import os
import wave

from dotenv import load_dotenv

import numpy as np

import pyaudio

from groq import Groq
from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice

### Set audio variables

# Load env variables
load_dotenv()
# Voice input
CHUNK_SIZE_INPUT = 1024  # Record in chunks of 1024 samples
SAMPLE_FORMAT_INPUT = pyaudio.paInt16  # 16 bits per sample
CHANNELS_INPUT = 2
FRAMES_PER_SECOND_INPUT = 44100  # Record at 44100 samples per second
START_BUFFER_SECONDS = 3
END_BUFFER_SECONDS = 2
chunks_per_second = int(FRAMES_PER_SECOND_INPUT / CHUNK_SIZE_INPUT)
#filename = os.path.dirname(__file__) + "/VoiceFiles/recorded_audio_stream.wav"
# Transcribe input
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq()
# Generate a text response
CHAT_PROMPT = "Você é o meu amigo brasileiro. " \
              "Estou tentando aprender português e estamos conversando em português." \
              "Suas respostas devem ter no máximo 3 frases."
system_prompt = dict(
    role="system",
    content=CHAT_PROMPT
)
# Initialize the chat history
chat_history = [system_prompt]
# Voice output
voice = GTTSVoice(language='pt', tld='com.br')
engine = GTTSEngine(voice=voice)
output_stream = TextToAudioStream(engine, language='pt')

# Set up
p = pyaudio.PyAudio()
# Reusable input/output streams
input_stream = p.open(format=SAMPLE_FORMAT_INPUT, channels=CHANNELS_INPUT, rate=FRAMES_PER_SECOND_INPUT, input=True)

# Start
print("Ola!")

while True:
    # Get user input
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    frames = []  # Initialize array to store frames
    max_audio = []  # Initialize array to store max audio values

    # Start streaming
    stream = p.open(format=SAMPLE_FORMAT_INPUT,
                    channels=CHANNELS_INPUT,
                    rate=FRAMES_PER_SECOND_INPUT,
                    input=True)

    while len(max_audio) < chunks_per_second * START_BUFFER_SECONDS \
            or np.max(max_audio[-int(chunks_per_second) * END_BUFFER_SECONDS:]) > 1000:
        # Store data in chunks
        data = input_stream.read(CHUNK_SIZE_INPUT)
        # Append max audio
        max_audio.append(np.max(np.frombuffer(data, dtype=np.int16)))
        # Append data to frames
        frames.append(data)

    # Write wave file
    waveBuffer = io.BytesIO()
    waveFile = wave.open(waveBuffer, 'wb')
    waveFile.setnchannels(CHANNELS_INPUT)
    waveFile.setsampwidth(p.get_sample_size(SAMPLE_FORMAT_INPUT))
    waveFile.setframerate(FRAMES_PER_SECOND_INPUT)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    # Seek the in-memory file to the start
    waveBuffer.seek(0)

    transcription = client.audio.transcriptions.create(
        file=('dummy_audio.wav', waveBuffer.read()),  # Dummy file for API request
        model="whisper-large-v3-turbo",
        prompt="Specify context or spelling.",
        response_format="json",
        language="pt",
        temperature=0.0
    )

    user_input = transcription.text

    # Append the user input to the chat history
    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_history,
        max_tokens=1024,
        temperature=1
    )

    edited_response = response.choices[0].message.content  # Make any format edits here

    # Append the response to the chat history
    chat_history.append({
      "role": "assistant",
      "content": edited_response
    })

    output_stream.feed(edited_response)
    output_stream.play()

# Close streams (if you break the loop)
input_stream.stop_stream()
input_stream.close()
p.terminate()
