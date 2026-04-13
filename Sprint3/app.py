"""
AI Social Media Caption Generator
Flask Web Application
Model   : ResNet101 + Bahdanau Attention + LSTM
Styling : Flan-T5-Base via HuggingFace Inference API
Extras  : Word Cloud · Attention Heatmap · Platform Comparison Chart
"""

import os
import io
import json
import uuid
import base64
import logging
import random
import pickle
import time
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── App Setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─── HuggingFace API ─────────────────────────────────────────────────────────
HF_API_KEY = "you token"   # your token here
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"   # change model here anytime
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type":  "application/json",
}

def _call_hf_api(prompt: str, retries: int = 2) -> str:
    if not HF_API_KEY:
        log.warning("HF_API_KEY not set.")
        return ""

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    for attempt in range(retries + 1):
        try:
            r      = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=40)
            result = r.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"].strip()
            elif "error" in result:
                log.warning("HF API error: %s", result["error"])
                return ""
        except Exception as e:
            log.warning("HF API attempt %d failed: %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(3)
    return ""

# ─── Globals ────────────────────────────────────────────────────────────────
caption_model      = None
tokenizer_data     = None
_resnet_extractor  = None
IMG_SIZE           = (224, 224)
MAX_CAPTION_LEN    = 40

import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_time  = tf.expand_dims(hidden, 1)
        score        = tf.nn.tanh(self.W1(features) + self.W2(hidden_time))
        attn_weights = tf.nn.softmax(self.V(score), axis=1)
        context      = tf.reduce_sum(attn_weights * features, axis=1)
        return context, attn_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


class AttentionCaptionModel(tf.keras.Model):
    def __init__(self, vocab_size=15000, embedding_dim=256,
                 units=512, feat_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.units         = units
        self.feat_dim      = feat_dim
        self.attn      = BahdanauAttention(units)
        self.feat_proj = tf.keras.layers.Dense(units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout   = tf.keras.layers.Dropout(0.3)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.fc1       = tf.keras.layers.Dense(units, activation="relu")
        self.fc2       = tf.keras.layers.Dense(vocab_size)

    def call(self, features, x, training=False):
        """features:(B,49,2048)  x:(B,T)"""
        features   = self.feat_proj(features)
        x_emb      = self.embedding(x)
        mean_feat  = tf.reduce_mean(features, axis=1)
        h, c       = mean_feat, tf.zeros_like(mean_feat)
        x_emb_list = tf.unstack(x_emb, axis=1)
        outputs    = []
        for x_t in x_emb_list:
            context, _ = self.attn(features, h)
            inp_t      = tf.concat([x_t, context], axis=-1)
            inp_t      = self.dropout(inp_t, training=training)
            out, [h, c] = self.lstm_cell(inp_t, states=[h, c], training=training)
            logits     = self.fc2(self.fc1(out))
            outputs.append(logits)
        return tf.stack(outputs, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocab_size": self.vocab_size,
                    "embedding_dim": self.embedding_dim,
                    "units": self.units, "feat_dim": self.feat_dim})
        return cfg

    @classmethod
    def from_config(cls, config):
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)


tf.keras.utils.get_custom_objects()["BahdanauAttention"]     = BahdanauAttention
tf.keras.utils.get_custom_objects()["AttentionCaptionModel"] = AttentionCaptionModel


def load_models():
    global caption_model, tokenizer_data

    weights_path = "model.weights.h5"
    keras_path   = "best_attention_model.keras"

    if not os.path.exists(weights_path) and not os.path.exists(keras_path):
        log.warning("No model file found → demo mode")
    else:
        log.info("Loading caption model …")
        try:
            caption_model = AttentionCaptionModel(
                vocab_size=15000, embedding_dim=256, units=512, feat_dim=2048)

            # Build with correct sequence length (MAX_LEN-1 = 33)
            dummy_feat = tf.zeros((1, 49, 2048))
            dummy_seq  = tf.zeros((1, 33), dtype=tf.int32)
            caption_model(dummy_feat, dummy_seq, training=False)
            log.info("Model architecture built ✓")

            if os.path.exists(weights_path):
                caption_model.load_weights(weights_path)
                log.info("Weights loaded from model.weights.h5 ✓")
            else:
                saved = tf.keras.models.load_model(
                    keras_path, compile=False,
                    custom_objects={
                        "BahdanauAttention":     BahdanauAttention,
                        "AttentionCaptionModel": AttentionCaptionModel,
                    })
                caption_model.set_weights(saved.get_weights())
                del saved
                log.info("Weights loaded from .keras fallback ✓")

        except Exception as e:
            log.error("Caption model load failed: %s", e)
            caption_model = None

    if os.path.exists("tokenizer.json"):
        try:
            with open("tokenizer.json") as f:
                raw = json.load(f)
            wi = raw.get("word_index", raw)
            tokenizer_data = {
                "word_index":  wi,
                "start_token": int(wi.get("<start>", wi.get("startseq", 3))),
                "end_token":   int(wi.get("<end>",   wi.get("endseq",   4))),
            }
            log.info("Tokenizer (json) loaded ✓  vocab=%d  start=%d  end=%d",
                     len(wi), tokenizer_data["start_token"], tokenizer_data["end_token"])
        except Exception as e:
            log.error("Tokenizer json failed: %s", e)

    elif os.path.exists("tokenizer.pkl"):
        try:
            with open("tokenizer.pkl", "rb") as f:
                tok = pickle.load(f)
            wi = tok.word_index if hasattr(tok, "word_index") else tok
            tokenizer_data = {
                "word_index":  wi,
                "start_token": int(wi.get("<start>", wi.get("startseq", 3))),
                "end_token":   int(wi.get("<end>",   wi.get("endseq",   4))),
            }
            log.info("Tokenizer (pkl) loaded ✓  vocab=%d  start=%d  end=%d",
                     len(wi), tokenizer_data["start_token"], tokenizer_data["end_token"])
        except Exception as e:
            log.error("Tokenizer pkl failed: %s", e)
    else:
        log.warning("No tokenizer file found")

    log.info("Warming up HuggingFace …")
    try:
        r = requests.post(
            HF_API_URL, headers=HF_HEADERS,
            json={
                "model": HF_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
            timeout=30,
        )
        if r.status_code == 200:
            log.info("HuggingFace ready ✓  model=%s", HF_MODEL)
        else:
            log.warning("HF warmup status: %d  body: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("HuggingFace warmup failed: %s", e)


def _get_resnet_extractor():
    global _resnet_extractor
    if _resnet_extractor is None:
        log.info("Building ResNet101 feature extractor …")
        base   = tf.keras.applications.ResNet101(include_top=False, weights="imagenet")
        feats  = tf.keras.layers.Reshape((49, 2048))(base.output)
        _resnet_extractor = tf.keras.Model(base.input, feats)
        log.info("ResNet101 extractor ready ✓")
    return _resnet_extractor


def preprocess_image(img_path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = tf.keras.applications.resnet.preprocess_input(np.array(img, dtype=np.float32))
    arr = np.expand_dims(arr, axis=0)
    return _get_resnet_extractor().predict(arr, verbose=0)


def generate_base_caption(img_features: np.ndarray) -> str:
    if caption_model is None or not tokenizer_data:
        return _demo_caption()
    try:
        wi          = tokenizer_data["word_index"]
        idx_to_word = {int(v): k for k, v in wi.items()}
        start_token = tokenizer_data["start_token"]
        end_token   = tokenizer_data["end_token"]
        skip        = {"", "<start>", "<end>", "<unk>", "startseq", "endseq", "start", "end", "unk"}

        caption_ids = [start_token]
        log.info("Decoding  start=%d  end=%d", start_token, end_token)

        for step in range(MAX_CAPTION_LEN):
            seq     = np.array([caption_ids], dtype=np.int32)
            preds   = caption_model(img_features, seq, training=False)
            next_id = int(np.argmax(preds[0, -1]))
            if next_id == end_token:
                log.info("End token at step %d", step)
                break
            caption_ids.append(next_id)

        words   = [idx_to_word.get(i, "") for i in caption_ids[1:]]
        caption = " ".join(w for w in words if w not in skip).strip()
        log.info("Caption: %s", caption)
        return caption if len(caption) > 3 else _demo_caption()
    except Exception as e:
        log.error("Caption generation error: %s", e)
        return _demo_caption()


def _demo_caption() -> str:
    return random.choice([
        "A group of people enjoying a beautiful sunset at the beach.",
        "A scenic mountain landscape covered in lush green trees.",
        "A cozy cafe interior with warm lighting and wooden furniture.",
        "A vibrant city street filled with people and colorful lights.",
        "A golden field of sunflowers stretching to the horizon.",
        "A chef preparing a delicious meal in a modern kitchen.",
        "Friends laughing together around a campfire under the stars.",
        "A serene lake reflecting the colors of a morning sky.",
    ])


PLATFORM_PROMPTS = {
    "instagram": (
        "Rewrite the following caption for Instagram. "
        "Add 2-3 relevant emojis at the start, make it upbeat and casual, "
        "and append 10 relevant hashtags at the end.\n"
        "Caption: {caption}\nInstagram caption:"
    ),
    "facebook": (
        "Rewrite the following caption for Facebook. "
        "Make it warm, conversational and friendly. "
        "Add an engaging question at the end to encourage comments.\n"
        "Caption: {caption}\nFacebook caption:"
    ),
    "linkedin": (
        "Rewrite the following caption for LinkedIn. "
        "Use a professional and insightful tone. "
        "End with 3-4 relevant professional hashtags.\n"
        "Caption: {caption}\nLinkedIn caption:"
    ),
}


def _call_hf_api(prompt: str, retries: int = 2) -> str:
    if not HF_API_KEY:
        log.warning("Skipping HF API call — HF_API_KEY is not set.")
        return ""

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    for attempt in range(retries + 1):
        try:
            r      = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=40)
            result = r.json()

            if isinstance(result, dict) and "error" in result:
                err_msg = str(result["error"]).lower()
                if "loading" in err_msg and attempt < retries:
                    wait = min(result.get("estimated_time", 20), 25)
                    log.info("Model loading, waiting %.0fs …", wait)
                    time.sleep(wait)
                    continue
                log.warning("HF API error: %s", result["error"])
                return ""

            if isinstance(result, dict) and "choices" in result:
                return result["choices"][0]["message"]["content"].strip()

            log.warning("Unexpected HF API response: %s", str(result)[:200])
            return ""

        except requests.exceptions.Timeout:
            log.warning("HF API timeout on attempt %d", attempt + 1)
            if attempt < retries:
                time.sleep(5)
        except Exception as e:
            log.warning("HF API attempt %d failed: %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(5)

    return ""


def format_caption(base_caption: str, platform: str) -> str:
    platform  = platform.lower().strip()
    prompt    = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["instagram"]).format(caption=base_caption)
    generated = _call_hf_api(prompt)
    if generated:
        log.info("Flan-T5 styled ✓ for %s", platform)
        return generated
    log.info("Falling back to rule-based formatter for %s", platform)
    return _rule_based_format(base_caption, platform)


_IG_TAGS = ["#photooftheday","#instagood","#beautiful","#love","#picoftheday",
            "#nature","#travel","#lifestyle","#happy","#photography",
            "#mood","#vibes","#aesthetic","#explore","#life"]
_EMOJI_MAP = {"beach":"🏖️","sunset":"🌅","mountain":"⛰️","city":"🌆","food":"🍽️",
              "friends":"👫","dog":"🐶","cat":"🐱","flower":"🌸","tree":"🌳",
              "sky":"🌤️","night":"🌙","people":"👥","smile":"😊","love":"❤️",
              "coffee":"☕","travel":"✈️","rain":"🌧️","snow":"❄️","lake":"🏞️"}


def _rule_based_format(caption: str, platform: str) -> str:
    if platform == "instagram":
        emojis = [e for w, e in _EMOJI_MAP.items() if w in caption.lower()] or ["✨","📸"]
        tags   = " ".join(random.sample(_IG_TAGS, min(10, len(_IG_TAGS))))
        return f"{' '.join(emojis[:3])} {caption}\n\n{tags}"
    if platform == "facebook":
        opener = random.choice(["Look at this amazing scene! ",
                                "Can you believe how stunning this is? ",
                                "This made my day and I had to share it! "])
        closer = random.choice(["\n\nWhat do you think? 💬", "\n\nTag someone who needs to see this! 👇"])
        return opener + caption + closer
    opener = random.choice(["Reflecting on this powerful moment:",
                            "This image captures an important truth:",
                            "Sharing this perspective with my network:"])
    cta = "\n\n💡 What does this mean to you?\n\n#Leadership #Inspiration #Growth #Mindset"
    return f"{opener}\n\n{caption}\n\nEvery image tells a story.{cta}"


def generate_wordcloud_b64(caption: str) -> str:
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        clean = caption.replace("#", "").replace("\n", " ")
        wc = WordCloud(width=500, height=240, background_color=None, mode="RGBA",
                       colormap="cool", max_words=40).generate(clean)
        fig, ax = plt.subplots(figsize=(5, 2.4), facecolor="none")
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        log.warning("Word cloud failed: %s", e); return ""


def generate_attention_heatmap_b64(img_path: str, img_features: np.ndarray) -> str:
    """
    AttentionCaptionModel is a subclassed tf.keras.Model, so it has no .inputs
    attribute.  We call the attention layer directly instead of trying to build a
    sub-model extractor (which only works for functional models).
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2
        from PIL import Image

        original   = np.array(Image.open(img_path).convert("RGB").resize(IMG_SIZE))
        attn_map   = None
        mode_label = "Image Saliency Map"

        if caption_model is not None:
            try:
                feats_tf   = tf.constant(img_features, dtype=tf.float32)
                proj_feats = caption_model.feat_proj(feats_tf)
                mean_feat  = tf.reduce_mean(proj_feats, axis=1)
                _, attn_w  = caption_model.attn(proj_feats, mean_feat)

                attn_flat = np.array(attn_w).reshape(-1)[:49]
                attn_map  = attn_flat.reshape(7, 7)
                mode_label = "Soft Attention Map"
                log.info("Attention extraction succeeded ✓")
            except Exception as e:
                log.warning("Attention extraction failed: %s", e)

        if attn_map is None:
            gray     = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            attn_map = cv2.GaussianBlur(
                np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32),
                (31, 31), 0
            )

        attn_norm    = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_resized = cv2.resize(attn_norm.astype(np.float32), IMG_SIZE)
        heatmap      = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )
        overlay   = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), facecolor="#0f1629")
        for ax, im, t in zip(axes, [original, overlay], ["Original", mode_label]):
            ax.imshow(im)
            ax.set_title(t, color="#94a3b8", fontsize=9, pad=6)
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f1629")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        log.warning("Heatmap failed: %s", e); return ""


def platform_comparison_data(base_caption: str) -> dict:
    metrics = {}
    for plat in ("instagram", "facebook", "linkedin"):
        styled = _rule_based_format(base_caption, plat)
        words  = styled.split()
        wc, cc, hc = len(words), len(styled), styled.count("#")
        avg_wl = sum(len(w.strip("#@.,!?")) for w in words) / max(wc, 1)
        targets = {"instagram": 150, "facebook": 300, "linkedin": 500}
        metrics[plat] = {
            "readability": max(0, min(100, int(100 - avg_wl * 5))),
            "length":      max(0, min(100, int(100 - abs(cc - targets[plat]) / targets[plat] * 100))),
            "engagement":  min(100, sum(1 for s in ["!","?","#","❤","😊","✨"] if s in styled) * 14),
            "hashtags":    min(100, hc * 10),
        }
    return metrics


def compute_analytics(caption: str, platform: str) -> dict:
    words = caption.split()
    wc, cc, hc = len(words), len(caption), caption.count("#")
    avg_wl       = sum(len(w.strip("#@.,!?")) for w in words) / max(wc, 1)
    readability  = max(0, min(100, int(100 - avg_wl * 5)))
    targets      = {"instagram": 150, "facebook": 300, "linkedin": 500}
    length_score = max(0, min(100, int(100 - abs(cc - targets.get(platform.lower(), 200)) / targets.get(platform.lower(), 200) * 100)))
    eng_score    = min(100, sum(1 for s in ["!","?","#","@","tag","comment","share","like","❤","😊","✨"] if s in caption) * 14)
    quality      = int(readability * 0.35 + length_score * 0.35 + eng_score * 0.30)
    pos_words    = {"beautiful","amazing","great","wonderful","happy","joy","love","stunning","incredible","fantastic","gorgeous","peaceful","vibrant"}
    neg_words    = {"sad","terrible","awful","bad","dark","broken","lost","hard","difficult"}
    lw           = set(w.lower().strip(".,!?") for w in words)
    pos_h, neg_h = len(lw & pos_words), len(lw & neg_words)
    if pos_h > neg_h:   sentiment, ss, sc = "Positive", min(100, 50+pos_h*10), "#10b981"
    elif neg_h > pos_h: sentiment, ss, sc = "Negative", max(0, 50-neg_h*10),   "#ef4444"
    else:               sentiment, ss, sc = "Neutral",  50,                     "#f59e0b"
    radar_profiles = {
        "instagram": {"professionalism":45,"engagement":90,"hashtags":95,"length":55},
        "facebook":  {"professionalism":60,"engagement":80,"hashtags":20,"length":70},
        "linkedin":  {"professionalism":95,"engagement":65,"hashtags":60,"length":85},
    }
    return {"word_count": wc, "char_count": cc, "hashtag_count": hc,
            "quality_score": quality, "readability": readability,
            "length_score": length_score, "engagement_score": eng_score,
            "sentiment": sentiment, "sentiment_score": ss, "sentiment_color": sc,
            "radar": radar_profiles.get(platform.lower(), radar_profiles["instagram"])}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file     = request.files["image"]
    platform = request.form.get("platform", "instagram")
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed."}), 400
    ext         = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path   = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)
    try:
        img_features   = preprocess_image(save_path)
        base_caption   = generate_base_caption(img_features)
        styled         = format_caption(base_caption, platform)
        analytics      = compute_analytics(styled, platform)
        wordcloud_b64  = generate_wordcloud_b64(styled)
        heatmap_b64    = generate_attention_heatmap_b64(save_path, img_features)
        platform_chart = platform_comparison_data(base_caption)
        return jsonify({
            "success": True, "image_url": f"/static/uploads/{unique_name}",
            "base_caption": base_caption, "caption": styled, "platform": platform,
            "analytics": analytics, "wordcloud": wordcloud_b64, "heatmap": heatmap_b64,
            "platform_chart": platform_chart, "styling_engine": "flan-t5-base (HuggingFace API)",
        })
    except Exception as exc:
        log.error("Pipeline error: %s", exc)
        return jsonify({"error": f"Caption generation failed: {str(exc)}"}), 500


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


with app.app_context():
    load_models()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)