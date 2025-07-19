from common.agent_functions import FUNCTION_DEFINITIONS
from common.prompt_templates import PROMPT_TEMPLATE
from datetime import datetime
import os
import glob

# Constants
VOICE_MODEL = "aura-2-thalia-en"
VOICE_NAME = "Thalia"
COMPANY_NAME = "Smart Sort"
VOICE_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# Audio Settings
USER_AUDIO_SAMPLE_RATE = 48000
USER_AUDIO_SECS_PER_CHUNK = 0.05
USER_AUDIO_SAMPLES_PER_CHUNK = round(USER_AUDIO_SAMPLE_RATE * USER_AUDIO_SECS_PER_CHUNK)

AGENT_AUDIO_SAMPLE_RATE = 16000
AGENT_AUDIO_BYTES_PER_SEC = 2 * AGENT_AUDIO_SAMPLE_RATE

AUDIO_SETTINGS = {
    "input": {
        "encoding": "linear16",
        "sample_rate": USER_AUDIO_SAMPLE_RATE,
    },
    "output": {
        "encoding": "linear16",
        "sample_rate": AGENT_AUDIO_SAMPLE_RATE,
        "container": "none",
    },
}

LISTEN_SETTINGS = {
    "provider": {
        "type": "deepgram",
        "model": "nova-3",
    }
}

THINK_SETTINGS = {
    "provider": {
        "type": "open_ai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
    },
    "prompt": PROMPT_TEMPLATE.format(current_date=datetime.now().strftime("%A, %B %d, %Y")),
    "functions": FUNCTION_DEFINITIONS,
}

SPEAK_SETTINGS = {
    "provider": {
        "type": "deepgram",
        "model": VOICE_MODEL,
    }
}

AGENT_SETTINGS = {
    "language": "en",
    "listen": LISTEN_SETTINGS,
    "think": THINK_SETTINGS,
    "speak": SPEAK_SETTINGS,
    "greeting": "",
}

SETTINGS = {"type": "Settings", "audio": AUDIO_SETTINGS, "agent": AGENT_SETTINGS}


def read_documentation_files(docs_dir):
    """Read all .mdx files in the specified directory and return their contents as a dictionary."""
    documentation = {}
    if not os.path.exists(docs_dir):
        return documentation

    mdx_files = glob.glob(os.path.join(docs_dir, "*.mdx"))
    for file_path in mdx_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                key = os.path.basename(file_path).replace(".mdx", "")
                documentation[key] = content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return documentation


class SmartSortAgent:
    def __init__(self, docs_dir="deepgram-docs/fern/docs"):
        self.voice_model = VOICE_MODEL
        self.voice_name = VOICE_NAME
        self.company = COMPANY_NAME
        self.docs_dir = docs_dir

        self.voice_agent_url = VOICE_AGENT_URL
        self.settings = SETTINGS
        self.user_audio_sample_rate = USER_AUDIO_SAMPLE_RATE
        self.user_audio_secs_per_chunk = USER_AUDIO_SECS_PER_CHUNK
        self.user_audio_samples_per_chunk = USER_AUDIO_SAMPLES_PER_CHUNK
        self.agent_audio_sample_rate = AGENT_AUDIO_SAMPLE_RATE
        self.agent_audio_bytes_per_sec = AGENT_AUDIO_BYTES_PER_SEC

        self.personality = (
            f"You are {self.voice_name}, a helpful and professional customer service representative "
            f"for {self.company}, a smart recycling and waste management company. Your role is to assist "
            f"customers with general inquiries about {self.company}'s services."
        )
        self.capabilities = "I can help you answer questions about Smart Sort."
        self.prompt = self.personality + "\n\n" + PROMPT_TEMPLATE.format(
            current_date=datetime.now().strftime("%A, %B %d, %Y")
        )
        self.first_message = (
            f"Hello! I'm {self.voice_name} from {self.company} customer service. "
            f"{self.capabilities} How can I help you today?"
        )

        self.settings["agent"]["speak"]["provider"]["model"] = self.voice_model
        self.settings["agent"]["think"]["prompt"] = self.prompt
        self.settings["agent"]["greeting"] = self.first_message
