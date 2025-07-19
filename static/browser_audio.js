const socket = io();

let audioContext = null;
let mediaStream = null;
let processor = null;
let sourceNode = null;
let isStreaming = false;

async function startMicStreaming() {
  if (isStreaming) return false; // Prevent multiple starts

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    processor = audioContext.createScriptProcessor(4096, 1, 1);
    sourceNode.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (event) => {
      if (!isStreaming) return;

      const inputBuffer = event.inputBuffer.getChannelData(0);
      const int16Buffer = new Int16Array(inputBuffer.length);

      for (let i = 0; i < inputBuffer.length; i++) {
        let s = Math.max(-1, Math.min(1, inputBuffer[i]));
        int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      socket.emit("audio_data", int16Buffer.buffer);
    };

    isStreaming = true;
    console.log("Mic streaming started");
    return true;
  } catch (err) {
    console.error("Error starting mic streaming:", err);
    return false;
  }
}

function stopMicStreaming() {
  if (!isStreaming) return;

  processor.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach(track => track.stop());
  audioContext.close();

  processor = null;
  sourceNode = null;
  mediaStream = null;
  audioContext = null;
  isStreaming = false;

  console.log("Mic streaming stopped");
}

// Log socket connection events
socket.on("connect", () => {
  console.log("[SocketIO] Connected to server");
});
socket.on("disconnect", () => {
  console.log("[SocketIO] Disconnected from server");
});
socket.on("error", (err) => {
  console.error("[SocketIO] Error:", err);
});

// Play audio output received from server
socket.on("audio_output", (data) => {
  console.log("[audio_output] Received", data);
  if (!data || !data.audio) {
    console.warn("[audio_output] No audio data received");
    return;
  }

  try {
    const sampleRate = data.sampleRate || 16000;
    const binary = atob(data.audio);
    if (binary.length === 0) {
      console.warn("[audio_output] Empty audio payload");
      return;
    }
    const buffer = new ArrayBuffer(binary.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) {
      view[i] = binary.charCodeAt(i);
    }
    const int16 = new Int16Array(buffer);
    if (int16.length === 0) {
      console.warn("[audio_output] Decoded Int16Array is empty");
      return;
    }
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }
    if (float32.length === 0) {
      console.warn("[audio_output] Decoded Float32Array is empty");
      return;
    }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = ctx.createBuffer(1, float32.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32);
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start();
    source.onended = () => ctx.close();
    console.log(`[audio_output] Played audio chunk (${float32.length} samples, sampleRate=${sampleRate})`);
  } catch (err) {
    console.error("[audio_output] Error processing audio output:", err);
  }
});

// Export for use in HTML
window.startMicStreaming = startMicStreaming;
window.stopMicStreaming = stopMicStreaming;
