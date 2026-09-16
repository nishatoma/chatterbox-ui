from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import gradio as gr
import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL: ChatterboxTurboTTS | None = None
MODEL_LOCK = threading.Lock()


CSS = """
:root {
  --cb-bg: #07131f;
  --cb-panel: #102333;
  --cb-border: #294153;
  --cb-text: #f4f1e8;
  --cb-muted: #aab4be;
  --cb-accent: #f0b44d;
}

.gradio-container {
  max-width: 1160px !important;
  margin: 0 auto !important;
}

body, .gradio-container {
  background: radial-gradient(circle at top, #102b40 0%, var(--cb-bg) 42%) !important;
  color: var(--cb-text) !important;
}

.hero {
  padding: 28px 30px;
  margin: 18px 0 16px;
  border: 1px solid var(--cb-border);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(16,35,51,.98), rgba(12,27,41,.94));
  box-shadow: 0 18px 60px rgba(0,0,0,.24);
}

.hero h1 { margin: 0 0 8px; font-size: 2rem; }
.hero p { margin: 0; color: var(--cb-muted); max-width: 760px; }

.status-card {
  border: 1px solid rgba(240,180,77,.35) !important;
  border-radius: 14px !important;
  background: rgba(240,180,77,.08) !important;
  padding: 10px 14px !important;
}

.panel {
  border: 1px solid var(--cb-border) !important;
  border-radius: 18px !important;
  background: rgba(16,35,51,.88) !important;
  padding: 18px !important;
}

.generate-btn {
  min-height: 50px !important;
  font-weight: 750 !important;
  font-size: 1rem !important;
  border-radius: 12px !important;
  color: #111820 !important;
  background: linear-gradient(135deg, #ffd681, var(--cb-accent)) !important;
  border: none !important;
}

.helper { color: var(--cb-muted); font-size: .92rem; }
.footer-note { text-align: center; color: var(--cb-muted); margin: 18px 0 30px; }
"""


def device_status() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"### GPU ready\n**{name}** · {total:.1f} GB VRAM · CUDA enabled"
    return "### GPU unavailable\nCUDA was not detected. Reinstall the CUDA-enabled PyTorch build."


def get_model(progress: gr.Progress) -> ChatterboxTurboTTS:
    global MODEL
    if MODEL is None:
        progress(0.05, desc="Loading Chatterbox Turbo (first run downloads the model)")
        MODEL = ChatterboxTurboTTS.from_pretrained(device="cuda")
    return MODEL


def split_text(text: str, max_chars: int = 320) -> list[str]:
    """Split narration into sentence-aware chunks for steadier generations."""
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                words = sentence.split()
                pieces: list[str] = []
                piece = ""
                for word in words:
                    candidate = f"{piece} {word}".strip()
                    if piece and len(candidate) > max_chars:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate
                if piece:
                    pieces.append(piece)
            else:
                pieces = [sentence]

            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate

        if current:
            chunks.append(current)

    return chunks


def generate_voice(
    text: str,
    reference_audio: str | None,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    pause_ms: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str]:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Add some text for Chatterbox to speak.")
    if not reference_audio:
        raise gr.Error("Record or upload a reference voice clip first.")
    if not torch.cuda.is_available():
        raise gr.Error("CUDA is unavailable. Verify your PyTorch CUDA installation.")

    chunks = split_text(text)
    if not chunks:
        raise gr.Error("The script did not contain any readable text.")
    if len(text) > 12_000:
        raise gr.Error("Keep one generation under 12,000 characters for reliability.")

    started = time.time()
    with MODEL_LOCK:
        model = get_model(progress)
        try:
            progress(0.12, desc="Analyzing your reference voice")
            model.prepare_conditionals(reference_audio)
        except AssertionError as exc:
            raise gr.Error("The reference recording must be longer than 5 seconds.") from exc
        except Exception as exc:
            raise gr.Error(f"Could not read the reference audio: {exc}") from exc

        generated: list[torch.Tensor] = []
        silence_samples = int(model.sr * max(0, pause_ms) / 1000)
        silence = torch.zeros((1, silence_samples))

        for index, chunk in enumerate(chunks, start=1):
            progress(
                0.12 + 0.82 * ((index - 1) / len(chunks)),
                desc=f"Generating section {index} of {len(chunks)}",
            )
            try:
                audio = model.generate(
                    chunk,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                ).cpu()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise gr.Error(
                    "The GPU ran out of memory. Close GPU-heavy apps and try a shorter script."
                ) from exc
            generated.append(audio)
            if index < len(chunks) and silence_samples:
                generated.append(silence)

        final_audio = torch.cat(generated, dim=1)
        output_path = OUTPUT_DIR / f"chatterbox-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        ta.save(str(output_path), final_audio, model.sr)

    duration = final_audio.shape[1] / model.sr
    elapsed = time.time() - started
    progress(1.0, desc="Finished")
    status = (
        f"✅ Generated **{duration:.1f} seconds** of audio in **{elapsed:.1f} seconds** "
        f"from **{len(chunks)} section{'s' if len(chunks) != 1 else ''}**."
    )
    return str(output_path), str(output_path), status


def clear_form():
    return "", None, None, None, "Ready for a new recording."


with gr.Blocks(title="Chatterbox Voice Studio") as app:
    gr.HTML(
        """
        <section class="hero">
          <h1>Chatterbox Voice Studio</h1>
          <p>Record a clean reference clip, paste your narration, and generate speech locally in your own voice.</p>
        </section>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, elem_classes=["panel"]):
            gr.Markdown("## 1 · Add your voice")
            reference = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                editable=True,
                label="Reference voice",
            )
            gr.Markdown(
                "Record or trim to **10–15 clean seconds** in a quiet room. M4A, MP3, and other uploads are converted to WAV automatically. "
                "For long recordings, use the scissors above to keep one clear section with little silence.",
                elem_classes=["helper"],
            )

        with gr.Column(scale=2, elem_classes=["panel"]):
            gr.Markdown(device_status(), elem_classes=["status-card"])
            gr.Markdown(
                "**Recommended reference**\n\n"
                "“Today I’m testing a local voice model using a clean recording of my natural speaking voice. "
                "I’m speaking clearly and at a comfortable pace.”"
            )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, elem_classes=["panel"]):
            gr.Markdown("## 2 · Write your script")
            script = gr.Textbox(
                label="Narration",
                placeholder="Paste the words you want your cloned voice to say…",
                lines=12,
                max_lines=24,
                buttons=["copy"],
            )
            gr.Markdown(
                "Turbo supports occasional tags such as `[laugh]`, `[chuckle]`, and `[cough]`. "
                "Long scripts are split automatically at sentence boundaries.",
                elem_classes=["helper"],
            )

        with gr.Column(scale=2, elem_classes=["panel"]):
            gr.Markdown("## 3 · Generate")
            with gr.Accordion("Advanced voice controls", open=False):
                temperature = gr.Slider(
                    0.3, 1.2, value=0.8, step=0.05,
                    label="Creativity / variation",
                    info="Lower is steadier; higher is more varied.",
                )
                top_p = gr.Slider(
                    0.5, 1.0, value=0.95, step=0.01,
                    label="Sampling range",
                )
                repetition = gr.Slider(
                    1.0, 1.5, value=1.2, step=0.05,
                    label="Repetition control",
                )
                pause = gr.Slider(
                    0, 1000, value=260, step=20,
                    label="Pause between generated sections (ms)",
                )

            generate = gr.Button("Generate my voice", variant="primary", elem_classes=["generate-btn"])
            status = gr.Markdown("Ready to generate.")
            output = gr.Audio(label="Generated voice", type="filepath")
            download = gr.DownloadButton("Download WAV", variant="secondary")
            clear = gr.Button("Clear", variant="secondary")

    gr.Markdown(
        "Generated audio stays on this computer. Chatterbox embeds an imperceptible AI-audio watermark. "
        "Only clone voices you own or have explicit permission to use.",
        elem_classes=["footer-note"],
    )

    generate.click(
        fn=generate_voice,
        inputs=[script, reference, temperature, top_p, repetition, pause],
        outputs=[output, download, status],
        concurrency_limit=1,
    )
    clear.click(
        fn=clear_form,
        outputs=[script, reference, output, download, status],
    )


if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch(
        inbrowser=True,
        server_name="127.0.0.1",
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )
