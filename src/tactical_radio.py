"""
Tactical Radio Dispatch Engine for Lucknow Geospatial AI.
Integrates Meta MMS (Massively Multilingual Speech) and ElevenLabs fallback
to convert Meta Llama 3.2 Vision tactical radio scripts into commanding, authentic audio broadcasts.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

AUDIO_OUTPUT_DIR = ROOT_DIR / "monitoring_results" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.meta_mms_radio import generate_tactical_radio_dispatch as generate_meta_dispatch


def generate_tactical_radio_dispatch(
    script_text: str,
    output_filename: str = "tactical_dispatch.wav",
    voice_name: str = "Adam"
) -> Path:
    """
    Generates tactical radio dispatch using Meta MMS open speech engine.
    If ELEVENLABS_API_KEY is present and desired, can optionally use ElevenLabs,
    otherwise uses 100% open-source Meta MMS.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if api_key and api_key.strip() != "your_elevenlabs_api_key_here":
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=api_key)
            out_path = AUDIO_OUTPUT_DIR / (output_filename if output_filename.endswith(".mp3") else f"{Path(output_filename).stem}.mp3")

            print(f"[TACTICAL RADIO] Generating ElevenLabs dispatch with voice '{voice_name}'...")
            audio = client.text_to_speech.convert(
                text=script_text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            with open(out_path, "wb") as f:
                for chunk in audio:
                    if chunk:
                        f.write(chunk)
            print(f"[TACTICAL RADIO] Successfully created audio dispatch at: {out_path}")
            return out_path
        except Exception as e:
            print(f"[TACTICAL RADIO] ElevenLabs note: {e}. Falling back to Meta MMS engine.")

    # Default: Use Meta MMS Open Speech Engine
    return generate_meta_dispatch(script_text=script_text, output_filename=output_filename)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sample_script = (
        "All units, urgent dispatch to Gomti River Floodplain Site LKO-ENC-20261023-468. "
        "Encroachment intercepted. Secure perimeter immediately. Over."
    )
    generate_tactical_radio_dispatch(sample_script)
