# CaptionAI — Social Media Caption Generator

AI-powered web application that generates platform-specific social media captions from images using ResNet101 + Soft Attention + LSTM.

---

## Project Structure

```
caption_app/
├── app.py                    # Flask backend (main entry point)
├── requirements.txt          # Python dependencies
├── best_attention_model.keras  ← Place your trained model here
├── tokenizer.json              ← Place your tokenizer here (optional)
│
├── templates/
│   └── index.html            # Main UI template
│
└── static/
    ├── css/
    │   └── style.css         # App stylesheet
    ├── js/
    │   └── app.js            # Frontend JavaScript + Chart.js
    └── uploads/              # Auto-created; stores uploaded images
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model
Place `best_attention_model.keras` in the project root (same level as `app.py`).

Optionally, place `tokenizer.json` alongside it for proper word decoding.
Expected format:
```json
{
  "word_index": { "<start>": 1, "a": 2, "dog": 3, ... },
  "start_token": 1
}
```

### 3. Run the app
```bash
python app.py
```

Then open: http://localhost:5000

### 4. Production deployment (Gunicorn)
```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## Model Architecture

The app expects a model accepting two inputs:
1. **Image array** — shape `(1, 224, 224, 3)`, preprocessed with ResNet101 channel-mean subtraction.
2. **Caption token sequence** — shape `(1, t)` — auto-regressive decoding loop.

If only an image input is detected, the app falls back to single-pass prediction.

If no model is found, the app runs in **demo mode** with pre-written sample captions.

---

## Platform Styles

| Platform | Style Applied |
|----------|---------------|
| Instagram | Emojis + 8-12 relevant hashtags |
| Facebook | Conversational opener + engagement CTA |
| LinkedIn | Professional tone + industry insight + hashtags |

---

## Visual Analytics

- **Caption Statistics** — word count, character count, hashtag count
- **Sentiment Analysis** — keyword-based Positive / Neutral / Negative score
- **Caption Quality Meter** — readability, length score, engagement score → composite score
- **Platform Style Radar** — Chart.js radar comparing Professionalism, Engagement, Hashtags, Length
- **Half-Donut Gauge** — canvas-drawn quality score visualization
