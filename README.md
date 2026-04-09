# Birthday Song Generator

Generates personalised birthday songs by replacing the sung name in a
template MP3 (e.g. one you made on UberDuck) with any name you like.
The result keeps the **exact same voice, melody, and energy** as the
original — only the name changes.

---

## How it works

1. **Transcribes** the template audio with OpenAI Whisper to find the
   original name and every timestamp where it is sung.
2. **Generates TTS** of the new name using Microsoft edge-tts — the same
   free cloud TTS that UberDuck-style tools use under the hood.
3. **Pitch-matches** the TTS clip to the original name's singing notes using
   `librosa`, so the new name rides the correct melody.
4. **Time-stretches** the clip to fit the original note duration perfectly.
5. **Crossfades** it back into the template, producing a seamless final MP3.

---

## Quick-start

```bash
# 1. Create & activate the virtual environment (already done if you see venv/)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate your first song  (uses the included template automatically)
python generate_birthday.py --name "Mohammed"
# → saved as:  happy_birthday_mohammed.mp3

python generate_birthday.py --name "Adam"
# → saved as:  happy_birthday_adam.mp3
```

---

## All options

```
python generate_birthday.py [OPTIONS]

  --name           REQUIRED. The name to sing.  e.g. --name "Sara"

  --template       Path to the template MP3.
                   Default: b2a4a162-2e45-444c-a23d-b713dc00dd4d.mp3

  --output         Output filename.
                   Default: happy_birthday_<name>.mp3

  --voice          edge-tts voice name (see list below).
                   Default: en-US-JennyNeural

  --whisper-model  tiny | base | small (default) | medium | large
                   Larger = more accurate name detection, but slower.

  --original-name  Override auto-detection if Whisper mishears the name.
                   e.g. --original-name "Ayesha"
```

---

## Recommended female voices

| Voice                  | Style              |
|------------------------|--------------------|
| en-US-JennyNeural      | Warm, natural (★ default) |
| en-US-AriaNeural       | Expressive         |
| en-US-MichelleNeural   | Friendly           |
| en-GB-SoniaNeural      | British            |
| en-AU-NatashaNeural    | Australian         |

Run `edge-tts --list-voices` to see the full list (200+ voices).

---

## Tips

- **First run downloads Whisper** (~240 MB for `small`). Subsequent runs use
  the cached model.
- If Whisper mishears the original name, pass it manually with
  `--original-name "CorrectName"`.
- Use `--whisper-model medium` for harder-to-transcribe audio.
- You can swap in any template MP3 with `--template path/to/my_song.mp3`.

---

## Dependencies (all free)

| Package        | Purpose                           |
|----------------|-----------------------------------|
| openai-whisper | Speech transcription + timestamps |
| edge-tts       | Microsoft TTS (free, no API key)  |
| librosa        | Pitch analysis & audio effects    |
| pydub          | Audio splicing & export           |
| soundfile      | Fast audio I/O                    |
| numpy / scipy  | Signal processing                 |
