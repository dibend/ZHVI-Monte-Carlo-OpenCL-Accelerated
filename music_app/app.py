import os
import numpy as np
import gradio as gr
import pyopencl as cl
import pyopencl.clrandom as clrandom
import soundfile as sf

MAX_TRACKS = 5
SAMPLE_RATE = 44100
DURATION_SEC = 5

# Cache context/queue
def get_opencl_context_queue():
    if hasattr(get_opencl_context_queue, "cache"):
        return get_opencl_context_queue.cache
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found")
    devices = platforms[0].get_devices()
    if not devices:
        raise RuntimeError("No OpenCL devices found")
    context = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(context)
    get_opencl_context_queue.cache = (context, queue)
    return context, queue

def generate_track(index):
    context, queue = get_opencl_context_queue()
    num_samples = SAMPLE_RATE * DURATION_SEC
    noise = clrandom.rand(queue, (num_samples,), dtype=np.float32).get() * 0.1
    t = np.arange(num_samples, dtype=np.float32) / SAMPLE_RATE
    freqs = np.random.uniform(220, 880, size=3)
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi))
    signal /= len(freqs)
    signal += noise
    signal /= np.max(np.abs(signal))
    os.makedirs('music_app/tracks', exist_ok=True)
    path = f'music_app/tracks/track_{index}.flac'
    sf.write(path, signal, SAMPLE_RATE, format='FLAC')
    return path

tracks = {}

def generate_single(i):
    path = generate_track(i)
    tracks[i] = path
    return path

def delete_single(i):
    path = tracks.pop(i, None)
    if path and os.path.exists(path):
        os.remove(path)
    return None

def generate_all(n):
    results = []
    for i in range(1, n+1):
        results.append(generate_single(i))
    for i in range(n+1, MAX_TRACKS+1):
        delete_single(i)
        results.append(None)
    return results

with gr.Blocks(title="OpenCL Music Generator") as demo:
    gr.Markdown("# OpenCL Music Generator")
    num_tracks = gr.Slider(1, MAX_TRACKS, value=3, step=1, label="Number of Tracks")
    generate_all_btn = gr.Button("Generate All Tracks")

    audio_components = []
    generate_buttons = []
    delete_buttons = []

    for i in range(1, MAX_TRACKS+1):
        with gr.Row():
            gen_btn = gr.Button(f"Generate Track {i}")
            del_btn = gr.Button(f"Delete Track {i}")
            audio = gr.Audio(label=f"Track {i}", interactive=False)
        generate_buttons.append(gen_btn)
        delete_buttons.append(del_btn)
        audio_components.append(audio)
        gen_btn.click(lambda idx=i: generate_single(idx), outputs=audio)
        del_btn.click(lambda idx=i: delete_single(idx), outputs=audio)

    generate_all_btn.click(generate_all, inputs=num_tracks, outputs=audio_components)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
