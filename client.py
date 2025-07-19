import os
import json
import time
import base64
import asyncio
import logging
import threading
import struct
import queue as std_queue
import uuid
import traceback

import pyaudio
import websockets
import janus
import requests

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# Add at the very top of the file, before any other imports
from gevent import monkey
monkey.patch_all()

# ---- Domain modules ----
from common.agent_functions import FUNCTION_MAP
from common.agent_templates import GenericCompanyAgent, AGENT_AUDIO_SAMPLE_RATE
from common.business_logic import (
    HARDCODED_USER,
    HARDCODED_APPOINTMENTS,
    HARDCODED_ORDERS,
)
from common.log_formatter import CustomFormatter

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BROWSER_INPUT_SAMPLE_RATE = 48000                # Browser mic capture rate
AGENT_OUTPUT_SAMPLE_RATE = AGENT_AUDIO_SAMPLE_RATE  # From template (e.g., 16000)
RAW_CHUNK_MAX_LATENCY_MS = 60
ASYNC_MODE = "threading"                         # Simpler integration
MIC_QUEUE_MAX = 50                               # Backpressure bound

try:
    import numpy as np
except ImportError:
    np = None

# ------------------------------------------------------------------
# Flask / SocketIO setup
# ------------------------------------------------------------------
app = Flask(__name__, static_folder="./static", static_url_path="/")
socketio = SocketIO(
    app,
    async_mode=ASYNC_MODE,
    cors_allowed_origins="*",
    ping_interval=25,
    ping_timeout=60,
    max_http_buffer_size=8_000_000,
)

logger = logging.getLogger("voice_agent")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter(socketio=socketio))
for h in list(logger.handlers):
    logger.removeHandler(h)
logger.addHandler(console_handler)


if os.environ.get("VOICE_AGENT_DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------
# Audio Helpers
# ------------------------------------------------------------------
def pcm16_resample_linear(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """
    Simple linear resampler (mono 16-bit) if numpy not available.
    """
    if src_rate == dst_rate:
        return pcm_bytes
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return pcm_bytes
    src = struct.unpack("<%dh" % sample_count, pcm_bytes)
    ratio = dst_rate / src_rate
    new_count = int(sample_count * ratio)
    out = []
    for i in range(new_count):
        pos = i / ratio
        i0 = int(pos)
        i1 = min(i0 + 1, sample_count - 1)
        frac = pos - i0
        val = int(src[i0] * (1 - frac) + src[i1] * frac)
        out.append(val)
    return struct.pack("<%dh" % len(out), *out)

def resample_if_needed(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm_bytes
    if np is not None:
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        new_length = int(len(arr) * dst_rate / src_rate)
        if new_length <= 0:
            return pcm_bytes
        new_idx = np.linspace(0, len(arr) - 1, new_length)
        new_arr = np.interp(new_idx, np.arange(len(arr)), arr).astype(np.int16)
        return new_arr.tobytes()
    return pcm16_resample_linear(pcm_bytes, src_rate, dst_rate)

# ------------------------------------------------------------------
# Voice Agent
# ------------------------------------------------------------------
class VoiceAgent:
    """
    Manages:
      - WebSocket connection to Deepgram
      - Sending user PCM16 frames
      - Receiving agent PCM16 frames & emitting to browser
      - Function call handling
    """
    def __init__(self, voice_model="aura-2-thalia-en", voice_name="", browser_audio=True):
        self.voice_model = voice_model
        self.voice_name = voice_name
        self.browser_audio = browser_audio
        self.browser_output = browser_audio

        self.agent_templates = GenericCompanyAgent()
        self.ws = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.is_running = False

        self.mic_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)

        self.audio = None
        self.stream = None
        self.input_device_id = None

        self.speaker = None
        self._tasks = []

    def set_loop(self, loop):
        self.loop = loop

    async def connect_deepgram(self) -> bool:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            logger.error("DEEPGRAM_API_KEY env var not present")
            return False

        settings = self.agent_templates.settings
        # Add a unique session_id for tracking
        session_id = str(uuid.uuid4())
        settings["session_id"] = session_id
        logger.info(f"Passing session_id to Deepgram: {session_id}")
        # If you choose to align to browser 48k, ensure settings reflect 48000 there.

        try:
            self.ws = await websockets.connect(
                self.agent_templates.voice_agent_url,
                extra_headers={"Authorization": f"Token {api_key}"},
                max_queue=None,
            )
            await self.ws.send(json.dumps(settings))
            logger.info("Connected to Deepgram Real-Time endpoint.")
            logger.debug(f"Deepgram settings used: {json.dumps(settings, indent=2)}")
            return True
        except Exception as e:
            logger.error(f"Deepgram connection failed: {e}\n{traceback.format_exc()}")
            return False

    async def start_local_microphone(self):
        self.audio = pyaudio.PyAudio()
        device_index = None
        if self.input_device_id and str(self.input_device_id).isdigit():
            device_index = int(self.input_device_id)

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.agent_templates.user_audio_sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.agent_templates.user_audio_samples_per_chunk,
            stream_callback=self._pyaudio_callback,
        )
        self.stream.start_stream()
        logger.info("Local microphone started.")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        if self.is_running and self.loop and not self.loop.is_closed():
            try:
                # Non-blocking: if full, drop oldest to maintain low latency
                q = self.mic_audio_queue
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                asyncio.run_coroutine_threadsafe(q.put(in_data), self.loop)
            except Exception as e:
                logger.error(f"Pyaudio callback queue error: {e}")
        return (in_data, pyaudio.paContinue)

    def cleanup_microphone(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
        self.stream = None
        self.audio = None

    async def audio_sender(self):
        logger.info("Audio sender started.")
        first = True
        try:
            while self.is_running:
                data = await self.mic_audio_queue.get()
                if not data:
                    continue
                if self.ws:
                    if first:
                        logger.debug(f"First outbound audio chunk {len(data)} bytes")
                        first = False
                    try:
                        await self.ws.send(data)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Deepgram WS closed while sending audio.")
                        break
        except asyncio.CancelledError:
            logger.info("Audio sender cancelled.")
        except Exception:
            logger.exception("Audio sender exception")

    async def audio_receiver_and_control(self):
        logger.info("Receiver started.")
        self.speaker = BrowserSpeaker(browser_output=self.browser_output)
        last_user_message_ts = None
        last_function_response_ts = None
        in_function_chain = False

        async with self.speaker:
            try:
                async for message in self.ws:
                    if isinstance(message, bytes):
                        await self.speaker.enqueue_pcm_chunk(message)
                        continue

                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON text frame ignored.")
                        continue

                    mtype = msg.get("type")
                    now = time.time()

                    if mtype == "Welcome":
                        logger.info(f"Session ID: {msg.get('session_id')}")
                    elif mtype == "ConversationText":
                        socketio.emit("conversation_update", msg)
                        role = msg.get("role")
                        if role == "user":
                            last_user_message_ts = now
                            in_function_chain = False
                        elif role == "assistant":
                            in_function_chain = False
                    elif mtype == "UserStartedSpeaking":
                        self.speaker.flush()
                    elif mtype == "FunctionCalling":
                        if in_function_chain and last_function_response_ts:
                            logger.info(
                                f"LLM Decision Latency (chain) "
                                f"{now - last_function_response_ts:.3f}s"
                            )
                        elif last_user_message_ts:
                            logger.info(
                                f"LLM Decision Latency (initial) "
                                f"{now - last_user_message_ts:.3f}s"
                            )
                            in_function_chain = True
                    elif mtype == "FunctionCallRequest":
                        await self._handle_function_call(msg)
                        last_function_response_ts = time.time()
                    elif mtype == "CloseConnection":
                        logger.info("Deepgram requested close.")
                        break
            except websockets.exceptions.ConnectionClosed:
                logger.info("Receiver: WebSocket closed.")
            except asyncio.CancelledError:
                logger.info("Receiver cancelled.")
            except Exception:
                logger.exception("Receiver error")

    async def _handle_function_call(self, msg):
        functions = msg.get("functions", [])
        if len(functions) != 1:
            logger.error("Only single function calls supported.")
            return
        fn_info = functions[0]
        name = fn_info.get("name")
        call_id = fn_info.get("id")
        raw_args = fn_info.get("arguments")

        if isinstance(raw_args, str):
            try:
                params = json.loads(raw_args or "{}")
            except Exception:
                params = {}
        elif isinstance(raw_args, dict):
            params = raw_args
        else:
            params = {}

        logger.info(f"FunctionCall: {name} args={params}")

        try:
            func = FUNCTION_MAP.get(name)
            if not func:
                raise ValueError(f"Unknown function '{name}'")

            if name in ("agent_filler", "end_call"):
                result = await func(self.ws, params)

                if name == "agent_filler":
                    inject = result["inject_message"]
                    function_response = result["function_response"]
                    await self._send_function_response(call_id, name, function_response)
                    await inject_agent_message(self.ws, inject)
                    return
                else:  # end_call
                    inject = result["inject_message"]
                    function_response = result["function_response"]
                    await self._send_function_response(call_id, name, function_response)
                    await wait_for_farewell_completion(self.ws, inject)
                    logger.info("Farewell completed. Closing WS.")
                    await close_websocket_with_timeout(self.ws)
                    self.is_running = False
                    return
            else:
                result = await func(params)

            await self._send_function_response(call_id, name, result)

        except Exception as e:
            logger.error(f"Function execution error: {e}")
            await self._send_function_response(call_id, name, {"error": str(e)})

    async def _send_function_response(self, call_id, name, payload):
        response = {
            "type": "FunctionCallResponse",
            "id": call_id,
            "name": name,
            "content": json.dumps(payload),
        }
        try:
            await self.ws.send(json.dumps(response))
            logger.info(f"Function response sent: {name}")
        except Exception as e:
            logger.error(f"Failed sending function response {name}: {e}")

    async def run(self):
        if not await self.connect_deepgram():
            return
        self.is_running = True

        if not self.browser_audio:
            await self.start_local_microphone()

        sender_task = asyncio.create_task(self.audio_sender(), name="audio_sender")
        recv_task = asyncio.create_task(
            self.audio_receiver_and_control(), name="audio_receiver"
        )
        self._tasks = [sender_task, recv_task]

        try:
            await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.is_running = False
            for t in self._tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self.cleanup_microphone()
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.close()
                except Exception:
                    pass
            logger.info("VoiceAgent stopped.")

    def stop(self):
        self.is_running = False
        if self.loop and not self.loop.is_closed():
            for task in asyncio.all_tasks(self.loop):
                task.cancel()

# ------------------------------------------------------------------
# Browser Speaker
# ------------------------------------------------------------------
class BrowserSpeaker:
    """
    Buffers agent PCM16 audio -> emits base64 to browser as chunks.
    """
    def __init__(self, browser_output=True):
        self.browser_output = browser_output
        self._queue: janus.Queue | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    async def __aenter__(self):
        self._queue = janus.Queue()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._thread_run, daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._queue:
            try:
                while not self._queue.sync_q.empty():
                    self._queue.sync_q.get_nowait()
            except std_queue.Empty:
                pass
        self._queue = None

    async def enqueue_pcm_chunk(self, pcm_bytes: bytes):
        if self._queue:
            await self._queue.async_q.put(pcm_bytes)

    def flush(self):
        if self._queue:
            try:
                while not self._queue.sync_q.empty():
                    self._queue.sync_q.get_nowait()
            except std_queue.Empty:
                pass

    def _thread_run(self):
        logger.info("BrowserSpeaker thread started.")
        while not self._stop_event.is_set():
            try:
                data: bytes = self._queue.sync_q.get(timeout=0.1)
            except std_queue.Empty:
                continue
            if not data:
                continue
            if self.browser_output:
                try:
                    b64 = base64.b64encode(data).decode("utf-8")
                    # Always emit with a valid sampleRate and optional timestamp
                    payload = {
                        "audio": b64,
                        "sampleRate": AGENT_OUTPUT_SAMPLE_RATE
                    }
                    # Optionally, if you want to pass a timestamp from the original chunk, you can add it here
                    # if hasattr(data, 'ts'):
                    #     payload["ts"] = data.ts
                    socketio.emit("audio_output", payload)
                    logger.debug(f"Emitted audio_output: {len(data)} bytes, sampleRate={AGENT_OUTPUT_SAMPLE_RATE}")
                except Exception:
                    logger.exception("Emit audio_output failed")

# ------------------------------------------------------------------
# Helper async functions
# ------------------------------------------------------------------
async def inject_agent_message(ws, inject_message):
    logger.info(f"Inject message: {inject_message}")
    await ws.send(json.dumps(inject_message))

async def close_websocket_with_timeout(ws, timeout=4):
    try:
        await asyncio.wait_for(ws.close(), timeout=timeout)
    except Exception:
        pass

async def wait_for_farewell_completion(ws, inject_message):
    """
    Simple farewell: send inject, wait for AgentStartedSpeaking + AgentAudioDone.
    """
    await inject_agent_message(ws, inject_message)
    spoke = False
    audio_done = False
    while not (spoke and audio_done):
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            break
        if isinstance(msg, bytes):
            continue
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue
        mtype = data.get("type")
        if mtype == "AgentStartedSpeaking":
            spoke = True
        if mtype == "AgentAudioDone":
            audio_done = True
    await asyncio.sleep(1.0)

# ------------------------------------------------------------------
# Global agent + start flag
# ------------------------------------------------------------------
voice_agent: VoiceAgent | None = None
_agent_starting = False

def _run_agent_thread():
    global voice_agent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if voice_agent:
        voice_agent.set_loop(loop)
        try:
            loop.run_until_complete(voice_agent.run())
        finally:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

# ------------------------------------------------------------------
# Socket.IO Handlers
# ------------------------------------------------------------------
@socketio.on("start_voice_agent")
def start_voice_agent(data=None):
    global voice_agent, _agent_starting
    if voice_agent and voice_agent.is_running:
        logger.info("VoiceAgent already running.")
        return
    if _agent_starting:
        logger.info("Start already in progress.")
        return
    _agent_starting = True
    try:
        logger.info(f"start_voice_agent payload: {data}")
        browser_audio = True if not data else data.get("browserAudio", True)
        voice_model = (data or {}).get("voiceModel", "aura-2-thalia-en")
        voice_name = (data or {}).get("voiceName", "")

        voice_agent = VoiceAgent(
            voice_model=voice_model,
            voice_name=voice_name,
            browser_audio=browser_audio,
        )
        if data:
            voice_agent.input_device_id = data.get("inputDeviceId")

        threading.Thread(target=_run_agent_thread, daemon=True).start()
        socketio.emit("agent_status", {"status": "starting"})
    finally:
        # small delay before allowing re-start attempts
        threading.Timer(1.0, lambda: globals().update(_agent_starting=False)).start()

@socketio.on("stop_voice_agent")
def stop_voice_agent():
    global voice_agent, _agent_starting
    _agent_starting = False
    if voice_agent:
        voice_agent.stop()
        voice_agent = None
    socketio.emit("agent_status", {"status": "stopped"})
    logger.info("VoiceAgent stop requested.")

@socketio.on("audio_data")
def audio_data(payload):
    """
    Expect payload = { audio: <base64 PCM16 mono>, sampleRate: <int> }
    """
    global voice_agent
    if not (voice_agent and voice_agent.is_running and voice_agent.browser_audio):
        return
    try:
        b64 = payload.get("audio")
        src_rate = int(payload.get("sampleRate", BROWSER_INPUT_SAMPLE_RATE))
        if not b64:
            return
        pcm = base64.b64decode(b64)

        # (Remove this block if Deepgram expects 48k already & your settings match)
        target_rate = voice_agent.agent_templates.user_audio_sample_rate
        if src_rate != target_rate:
            pcm = resample_if_needed(pcm, src_rate, target_rate)

        if voice_agent.loop and not voice_agent.loop.is_closed():
            q = voice_agent.mic_audio_queue
            if q.full():
                # drop oldest to keep latency bounded
                try:
                    voice_agent.loop.call_soon_threadsafe(lambda: q.get_nowait())
                except Exception:
                    pass
            asyncio.run_coroutine_threadsafe(q.put(pcm), voice_agent.loop)
    except Exception as e:
        logger.error(f"audio_data processing error: {e}")

# ------------------------------------------------------------------
# HTTP Endpoints
# ------------------------------------------------------------------
@app.route("/")
def index():
    sample_data = [{
        "Customer": HARDCODED_USER["name"],
        "ID": HARDCODED_USER["customer_id"],
        "Username": HARDCODED_USER["username"],
        "Phone": HARDCODED_USER["phone"],
        "Email": HARDCODED_USER["email"],
        "RewardPoints": HARDCODED_USER["reward_points"],
        "Appointments": [
            {"Service": a["service"], "Date": a["date"], "Status": a["status"]}
            for a in HARDCODED_APPOINTMENTS
        ],
        "Orders": [
            {
                "ID": o["id"],
                "# Items": o["items"],
                "Total": o["total"],
                "Status": o["status"],
                "Date": o["date"],
            }
            for o in HARDCODED_ORDERS
        ],
    }]
    return render_template("index.html", sample_data=sample_data)

@app.route("/tts-models")
def tts_models():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return jsonify({"error": "DEEPGRAM_API_KEY not set"}), 500
    try:
        resp = requests.get(
            "https://api.deepgram.com/v1/models",
            headers={"Authorization": f"Token {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), 500
        data = resp.json()
        models = []
        for m in data.get("tts", []):
            if m.get("architecture") == "aura-2":
                lang = (m.get("languages") or ["en"])[0]
                md = m.get("metadata", {})
                models.append({
                    "name": m.get("canonical_name", m.get("name")),
                    "display_name": m.get("name"),
                    "language": lang,
                    "accent": md.get("accent", ""),
                    "tags": ", ".join(md.get("tags", [])),
                })
        return jsonify({"models": models})
    except requests.RequestException as e:
        logger.error(f"TTS fetch network error: {e}")
        return jsonify({"error": "network_error"}), 500
    except Exception as e:
        logger.error(f"TTS fetch error: {e}")
        return jsonify({"error": str(e)}), 500

USER_CONTEXT = None

@app.route("/api/user-data", methods=["POST"])
def set_user_context():
    global USER_CONTEXT
    USER_CONTEXT = request.json
    logger.info(f"User context updated: {USER_CONTEXT}")
    return jsonify({"status": "ok"})

# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Voice Agent Backend Starting")
    print("=" * 60)
    print("Open http://127.0.0.1:3000 in your browser.")
    print("=" * 60 + "\n")
    socketio.run(app, host="127.0.0.1", port=3000, debug=True, use_reloader=False)
