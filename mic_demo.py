#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mic_demo.py – Live Echo Cancellation Demo (Microphone → Speaker) with delay alignment
------------------------------------------------------------------------------------
Uses `StreamingLMSFilter` to cancel speaker‑to‑mic echo. Replaces `aplay` with
a low‑latency SoundDevice output stream to avoid underruns, and adds a fixed
reference delay to align the monitor signal with the mic echo path.

Toggle AEC on/off with “m”.
"""

import os
import sys
import subprocess
import venv
import threading
import queue
import termios
import tty

# ─── VENV BOOTSTRAP ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
VENV_DIR   = os.path.join(SCRIPT_DIR, '.venv')
PY_BIN     = os.path.join(VENV_DIR, 'bin', 'python')
PIP_BIN    = os.path.join(VENV_DIR, 'bin', 'pip')

def ensure_venv_and_reexec():
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        if not os.path.isdir(os.path.join(VENV_DIR, 'bin')):
            print("🚀 Creating virtual environment…")
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        print("📦 Installing/upgrading pip, numpy, sounddevice…")
        subprocess.check_call([PIP_BIN, 'install', '--upgrade', 'pip'])
        subprocess.check_call([PIP_BIN, 'install', 'numpy', 'sounddevice'])
        print("🔄 Re-launching inside virtual environment…")
        os.execv(PY_BIN, [PY_BIN] + sys.argv)

ensure_venv_and_reexec()

# ─── IMPORTS & LMS FILTER ─────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
from collections import deque
from lms import StreamingLMSFilter

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
FS                = 48000      # sample rate
BLOCK             = 1024       # block size
MU                = 5e-4       # LMS step size
BAR_WIDTH         = 40
MAX_RMS           = 0.05
REF_DELAY_BLOCKS  = 2          # delay reference by this many blocks

aec_enabled       = True
monitor_buffer    = deque(maxlen=FS)     # ~1 s of monitor samples
frames_queue      = queue.Queue(maxsize=128)

# streaming LMS filter: BLOCK taps → matches block‐wise processing
lms_filterer = StreamingLMSFilter(num_taps=BLOCK, mu=MU, safe=True)

# ─── UTILS ───────────────────────────────────────────────────────────────────
def print_volume_bar(rms: float):
    frac   = min(rms / MAX_RMS, 1.0)
    filled = int(frac * BAR_WIDTH)
    empty  = BAR_WIDTH - filled
    bar    = '▇' * filled + '-' * empty
    mode   = 'AEC' if aec_enabled else 'PASS'
    sys.stdout.write(f"\rMic RMS |{bar}| {rms:.4f}  Mode={mode}")
    sys.stdout.flush()

# ─── AUDIO CALLBACKS ─────────────────────────────────────────────────────────
def monitor_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[Monitor status] {status}")
    monitor_buffer.extend(indata[:, 0].tolist())

def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[Mic status] {status}")

    mic = indata[:, 0].astype(np.float32)
    rms = float(np.sqrt(np.mean(mic**2)))
    print_volume_bar(rms)

    # determine output block
    if not aec_enabled or len(monitor_buffer) < (REF_DELAY_BLOCKS + 1) * BLOCK:
        out = mic
    else:
        # align reference by REF_DELAY_BLOCKS
        start = - (REF_DELAY_BLOCKS + 1) * BLOCK
        end   = - REF_DELAY_BLOCKS * BLOCK
        ref_block = np.array(list(monitor_buffer)[start:end], dtype=np.float32)
        out = lms_filterer.process_block(ref_block, mic)

    # convert to int16 and enqueue
    int_block = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    try:
        frames_queue.put_nowait(int_block)
    except queue.Full:
        pass  # drop if queue is full

def output_callback(outdata, frames, time_info, status):
    if status.output_underflow:
        print(f"\nOutput underflow at {time_info.outputBufferDacTime:.3f}")
    try:
        block = frames_queue.get_nowait()
        outdata[:] = block.reshape(-1, 1)
    except queue.Empty:
        outdata.fill(0)

# ─── KEY LISTENER ────────────────────────────────────────────────────────────
def key_listener():
    global aec_enabled
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch.lower() == 'm':
                aec_enabled = not aec_enabled
                print(f"\n>> Mode toggled to {'AEC' if aec_enabled else 'passthrough'} <<")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🔊 Starting Live Echo Canceller with delay alignment")
    print("   Press 'm' to toggle AEC on/off, Ctrl+C to quit")

    # start monitor (loopback) capture
    mon_stream = sd.InputStream(
        samplerate=FS, blocksize=BLOCK, channels=1, dtype='float32',
        callback=monitor_callback
    )
    mon_stream.start()

    # start mic capture
    mic_stream = sd.InputStream(
        samplerate=FS, blocksize=BLOCK, channels=1, dtype='float32',
        callback=mic_callback
    )
    mic_stream.start()

    # start speaker output
    out_stream = sd.OutputStream(
        samplerate=FS, blocksize=BLOCK, channels=1, dtype='int16',
        callback=output_callback
    )
    out_stream.start()

    # start key listener
    threading.Thread(target=key_listener, daemon=True).start()

    # run until interrupted
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        mic_stream.stop(); mic_stream.close()
        mon_stream.stop(); mon_stream.close()
        out_stream.stop(); out_stream.close()
        print("\n🛑 Stopped.")
