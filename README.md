# Chatterbox Voice Studio for Windows

A friendly local browser interface for [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox). Record or upload your own voice, paste a script, and generate downloadable WAV narration on your NVIDIA GPU.

The interface runs locally at `http://127.0.0.1:7860`. It does not create a public Gradio share link.

## Features

- Record a reference clip with your microphone or upload existing audio.
- Automatically convert M4A and MP3 reference files to WAV.
- Trim long recordings in the browser.
- Generate with Chatterbox Turbo on CUDA.
- Split long narration into sentence-aware sections.
- Adjust temperature, top-p, repetition penalty, and pauses.
- Preview and download generated WAV files.
- Store generations locally in `outputs/`.

## Tested configuration

- Windows 11
- Python 3.11
- NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM
- NVIDIA driver 560.94 / CUDA 12.6 compatibility
- PyTorch 2.6.0 with the CUDA 12.6 wheel
- Gradio 6.8

Other recent NVIDIA GPUs should work, although generation speed and available script length depend on VRAM.

## Installation

Install [Python 3.11](https://www.python.org/downloads/) and [Git](https://git-scm.com/download/win), then open PowerShell:

```powershell
git clone https://github.com/nishatoma/chatterbox-ui.git
cd chatterbox-ui

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install git+https://github.com/resemble-ai/chatterbox.git
```

Verify CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see `CUDA: True` and your NVIDIA GPU name.

## Start the interface

Double-click `START_UI.bat`, or run:

```powershell
.\.venv\Scripts\python.exe app.py
```

The interface opens automatically. If it does not, visit:

```text
http://127.0.0.1:7860
```

Keep the terminal window open while using the interface.

## Getting a natural voice

Use a clean 10–15 second reference clip. Speak with the pace and energy you want the generated narration to copy. Avoid background music, long silence, clipping, echo, and heavy noise reduction.

A useful starting preset for conversational narration is:

| Control | Value |
|---|---:|
| Creativity / temperature | `0.85` |
| Sampling range / top-p | `0.97` |
| Repetition control | `1.10` |
| Section pause | `150 ms` |

Group related sentences into ordinary paragraphs. A blank line creates a new generation section, so placing every sentence on its own paragraph can produce start-stop delivery.

## Reference-audio notes

- The reference must be longer than five seconds.
- M4A and MP3 uploads are converted to WAV by the interface.
- For long recordings, trim one clean 10–15 second section using the waveform editor.
- Chatterbox Turbo analyzes only a short conditioning window; uploading several minutes does not improve the clone.

## Generated files

WAV files are saved in:

```text
outputs\
```

The folder is ignored by Git.

## Responsible use

Only clone a voice you own or have explicit permission to use. Chatterbox embeds an imperceptible PerTh watermark in generated audio.

## Credits

This project is a local UI wrapper around [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI. It is not affiliated with or endorsed by Resemble AI.
