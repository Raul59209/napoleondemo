"""
app_demo.py — Napoleon Medical Pipeline
========================================
Flow:
  Mic recording OR file upload  →  click "Lancer"
  The pipeline runs fully automatically:
    1. Transcription  (Kyutai STT 1B en/fr, via transformers — one-shot, NOT live streaming)
    2. Correction
    3. Hallucination check  (local)
    4. Medical review  (LLM)
    5. Diarizatiion
    6. DPI enrichment  (LLM call 1)
    7. CR generation   (LLM call 2)
    8. PDF generation  (reportlab)

  Tabs are read-only result viewers — no more button-per-step.

NOTE ON STREAMING: Kyutai STT is architecturally a streaming model, but this
app does NOT show live word-by-word captions while recording. st.audio_input
only returns a complete recording after the user clicks stop — same as a
file upload, just from a mic instead of a file. The transcript is then sent
to Kyutai in ONE shot, same call pattern as a batch model. True live
streaming would require a custom JS component + WebSocket connection to
Kyutai's Rust server, which is a separate, much larger build (see chat).

NOTE ON LONG/REPEAT AUDIO: benchmark testing found Kyutai 1B can fall into
repetition loops ("oui oui oui...") on some longer recordings (2 of 6 test
files in one run had WER ~0.77 due to this). detect_hallucination() below
catches this same way it already did for the previous STT engine — it's not
new protection added for this swap, but it's the safety net this issue
actually needs given Kyutai's known failure mode.
"""

import io
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import torch
import torchaudio
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

KYUTAI_MODEL_ID = "kyutai/stt-1b-en_fr-trfs"   # bilingual en/fr
KYUTAI_TARGET_SR = 24000                        # Kyutai STT expects 24kHz audio

@st.cache_resource
def load_kyutai():
    """
    Loaded once per Streamlit server process (not per-run) via st.cache_resource —
    this model takes real time to load, and re-loading it on every pipeline run
    would make every consultation noticeably slower for no reason.

    NOTE: deliberately NOT using device_map=device here. device_map is meant
    for splitting a model across MULTIPLE devices/GPUs; for a single device
    it's simpler and more robust to load normally then .to(device). Passing
    a plain device string to device_map raised a ValueError inside
    accelerate's check_and_set_device_map on the transformers/accelerate
    versions Streamlit Cloud installed — switching to .to(device) avoids
    that code path entirely.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = KyutaiSpeechToTextProcessor.from_pretrained(KYUTAI_MODEL_ID)
    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
        KYUTAI_MODEL_ID, torch_dtype="auto"
    )
    model = model.to(device)
    return processor, model, device

kyutai_processor, kyutai_model, kyutai_device = load_kyutai()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Napoleon — Pipeline Médical",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #F7F6F2; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #0D1B3E; }
    .napoleon-header {
        background: linear-gradient(135deg, #0D1B3E 0%, #1a2f5e 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem;
        display: flex; align-items: center; gap: 1.5rem;
    }
    .napoleon-header h1 { color: white !important; font-size: 2.2rem; margin: 0; font-family: 'DM Serif Display', serif; }
    .napoleon-header p  { color: #A0B4CC; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .napoleon-badge { background: #028090; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .step-card { background: white; border-radius: 12px; padding: 1.5rem; border: 1px solid #E8E6E0; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .metric-box { background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px; padding: 0.8rem 1.2rem; flex: 1; text-align: center; }
    .metric-box .value { font-size: 1.6rem; font-weight: 600; color: #028090; font-family: 'DM Serif Display', serif; }
    .metric-box .label { font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    .alert-ok    { background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 8px; padding: 0.8rem 1.2rem; color: #065F46; font-weight: 500; }
    .alert-warn  { background: #FFF7ED; border: 1px solid #FCD34D; border-radius: 8px; padding: 0.8rem 1.2rem; color: #92400E; font-weight: 500; }
    .alert-error { background: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 8px; padding: 0.8rem 1.2rem; color: #991B1B; font-weight: 500; }
    .json-section { background: white; border-radius: 12px; border: 1px solid #E8E6E0; margin-bottom: 1rem; overflow: hidden; }
    .json-section-header { background: #0D1B3E; color: white; padding: 0.8rem 1.2rem; font-weight: 600; font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: white; border-radius: 12px; padding: 6px; border: 1px solid #E8E6E0; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 500; color: #64748B; }
    .stTabs [aria-selected="true"] { background: #0D1B3E !important; color: white !important; }
    .stButton > button { background: #028090; color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 600; font-family: 'DM Sans', sans-serif; }
    .stButton > button:hover { background: #026070; }
    .stDownloadButton > button { background: #0D1B3E; color: white; border: none; border-radius: 8px; font-weight: 600; }
    div[data-testid="stFileUploader"] { background: white; border-radius: 12px; border: 2px dashed #CBD5E1; padding: 1rem; }
    .pipeline-step { display:flex; align-items:center; gap:0.6rem; padding:0.35rem 0; font-size:0.9rem; color:#475569; }
    .pipeline-step .done { color:#059669; font-weight:700; }
    .pipeline-step .spin { color:#028090; font-weight:700; }
    .pipeline-step .wait { color:#CBD5E1; }
</style>
""", unsafe_allow_html=True)


# ── API key ───────────────────────────────────────────────────────────────────
def ensure_api_key() -> str:
    key = os.getenv("SCW_API_KEY")
    if key:
        return key
    st.markdown('<div class="napoleon-header"><div style="font-size:2.5rem">🩺</div><div><h1>Napoleon</h1><p>Configuration initiale requise</p></div></div>', unsafe_allow_html=True)
    st.markdown("### Clé API Scaleway")
    st.markdown("Napoleon utilise les GPUs Scaleway. [Obtenir une clé →](https://console.scaleway.com/iam/api-keys)")
    key_input = st.text_input("Clé API Scaleway", type="password", placeholder="scw-...")
    if st.button("Enregistrer et continuer"):
        if key_input.strip():
            with open(Path(__file__).parent / ".env", "a") as f:
                f.write(f"\nSCW_API_KEY={key_input.strip()}\n")
            st.success("Clé enregistrée. Rechargement...")
            st.rerun()
        else:
            st.error("Veuillez entrer une clé valide.")
    st.stop()

api_key = ensure_api_key()

from openai import OpenAI
scw_client = OpenAI(base_url="https://api.scaleway.ai/v1", api_key=api_key)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="napoleon-header">
    <div style="font-size:2.5rem">🩺</div>
    <div><h1>Napoleon</h1><p>Pipeline de traitement audio médical — transcription, extraction, rapport</p></div>
    <div style="margin-left:auto"><span class="napoleon-badge">Demo</span></div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "transcript":           None,
    "hallucination_ok":     None,
    "hallucination_reason": None,
    "review":               None,
    "audio_filename":       None,
    "enriched_dpi":         None,
    "cr":                   None,
    "pdf_bytes":            None,
    "pipeline_done":        False,
    "word_count":           0,
    "stt_time":             0.0,
    "llm_time":             0.0,
    "total_time":           0.0,
    "diarization":          None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def detect_hallucination(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < 10:
        return True, "Transcription vide ou trop courte."
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if len(sentences) < 3:
        return False, "OK"
    substantial = [s for s in sentences if len(s.split()) >= 4]
    if len(substantial) < 3:
        return False, "OK"
    counts = Counter(substantial)
    most_common, freq = counts.most_common(1)[0]
    if freq > 8:
        return True, f"Boucle détectée : \"{most_common[:60]}\" répété {freq} fois."
    if freq / len(substantial) > 0.4 and freq > 4:
        return True, f"Contenu répétitif : \"{most_common[:60]}\" ({freq}/{len(substantial)} phrases)."
    unique_chars = len(" ".join(set(substantial)))
    total_chars  = sum(len(s) for s in substantial)
    if total_chars > 500 and unique_chars / total_chars < 0.15:
        return True, "Ratio contenu unique/total très faible — hallucination probable."
    return False, "Aucune boucle détectée."


def transcribe_audio(audio_bytes: bytes, filename: str) -> tuple[str, int]:
    """
    Kyutai STT 1B (en/fr) via transformers — ONE-SHOT batch call.
    Works identically whether audio_bytes came from a mic recording
    (st.audio_input) or an uploaded file — both arrive as bytes here.

    Loads the bytes through torchaudio (works for wav/m4a/mp3/flac/ogg via
    its ffmpeg backend), downmixes to mono, resamples to 24kHz (Kyutai's
    expected input rate), then runs it through the processor/model.

    IMPORTANT: audio must be passed as the explicit `audio=` keyword to
    the processor. KyutaiSpeechToTextProcessor.__call__ has signature
    (images=None, text=None, videos=None, audio=None, ...) — passing the
    array positionally lands it in the `images` slot instead, producing an
    EMPTY BatchFeature and a downstream crash in model.generate() reading
    .shape on None. This bug cost real debugging time earlier — do not
    revert to positional.
    """
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
        if waveform.shape[0] > 1:  # downmix stereo to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != KYUTAI_TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, KYUTAI_TARGET_SR)
        audio_array = waveform.squeeze(0).numpy()

        inputs = kyutai_processor(audio=audio_array, return_tensors="pt")
        inputs = inputs.to(kyutai_device)

        with torch.no_grad():
            output_tokens = kyutai_model.generate(**inputs)

        text = kyutai_processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        text = text.strip()
        return text, len(text.split())
    except Exception as e:
        raise RuntimeError(f"Kyutai STT failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)



def call_llm(prompt: str, max_tokens: int = 4000) -> dict:
    try:
        response = scw_client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Strip anything after the last closing brace (extra text from LLM)
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"JSON invalide : {e}", "raw": raw[:500]}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATOR  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf(enriched_dpi: dict, cr: dict, filename_stem: str) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
    except ImportError:
        return None

    NAVY   = colors.HexColor("#0D1B3E")
    BORDER = colors.HexColor("#E8E6E0")
    GREY   = colors.HexColor("#64748B")
    WHITE  = colors.white
    W      = 160*mm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=15*mm, bottomMargin=15*mm,
                            leftMargin=15*mm, rightMargin=15*mm)
    base = getSampleStyleSheet()

    def sty(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    S_SEC  = sty("Sec",  fontSize=9,  fontName="Helvetica-Bold", textColor=WHITE)
    S_LBL  = sty("Lbl",  fontSize=8,  fontName="Helvetica-Bold", textColor=NAVY)
    S_BODY = sty("Body", fontSize=8,  fontName="Helvetica",
                 textColor=colors.HexColor("#1e293b"), leading=11)
    S_TTL  = sty("Ttl",  fontSize=13, fontName="Helvetica-Bold",
                 textColor=WHITE, alignment=TA_CENTER)
    S_SUB  = sty("Sub",  fontSize=8,  fontName="Helvetica",
                 textColor=colors.HexColor("#A0B4CC"), alignment=TA_CENTER)
    S_FOOT = sty("Foot", fontSize=6,  fontName="Helvetica",
                 textColor=GREY, alignment=TA_CENTER)

    def sec(label):
        return ([Paragraph(f"  {label.upper()}", S_SEC), ""], "sec")

    def row(label, value, shade=False):
        bg = colors.HexColor("#F8FAFC") if shade else WHITE
        return ([Paragraph(label, S_LBL), Paragraph(str(value or "—"), S_BODY)], bg)

    def make_table(entries):
        rows = [r for r, _ in entries]
        t = Table(rows, colWidths=[45*mm, 115*mm])
        cmds = [
            ("ALIGN",        (0,0),(-1,-1),"LEFT"),
            ("VALIGN",       (0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",  (0,0),(-1,-1),5),
            ("RIGHTPADDING", (0,0),(-1,-1),5),
            ("TOPPADDING",   (0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("GRID",         (0,0),(-1,-1),0.3,BORDER),
        ]
        for i, (_, meta) in enumerate(entries):
            if meta == "sec":
                cmds += [("SPAN",(0,i),(-1,i)),
                         ("BACKGROUND",(0,i),(-1,i),NAVY),
                         ("TOPPADDING",(0,i),(-1,i),5),
                         ("BOTTOMPADDING",(0,i),(-1,i),5)]
            elif meta != WHITE:
                cmds.append(("BACKGROUND",(0,i),(-1,i),meta))
        t.setStyle(TableStyle(cmds))
        return t

    def sv(obj, *keys, default="—"):
        for k in keys:
            if not isinstance(obj, dict): return default
            obj = obj.get(k)
            if obj is None: return default
        v = str(obj).strip()
        return v if v else default

    def ls(obj, key):
        return (obj or {}).get(key) or []

    # ── Pull data from flat schema ─────────────────────────────────────────────
    dpi_textuel       = (cr or {}).get("dpi_textuel") or ""
    cr_textuel        = (cr or {}).get("cr_textuel") or ""
    prescs            = (cr or {}).get("prescription_lines") or []

    motif             = enriched_dpi.get("motif_de_consultation") or ""
    antecedents       = enriched_dpi.get("antecedents") or {}
    mode_vie          = enriched_dpi.get("mode_de_vie") or {}
    trts_hab          = enriched_dpi.get("traitements_habituels") or []
    allergies         = enriched_dpi.get("allergies") or []
    interrogatoire_d  = enriched_dpi.get("interrogatoire") or {}
    examen_d          = enriched_dpi.get("examen_clinique") or {}
    conclusion        = enriched_dpi.get("conclusion") or {}

    story = []

    # ── Banner ────────────────────────────────────────────────────────────────
    banner = Table([
        [Paragraph("🩺  NAPOLEON — COMPTE-RENDU DE CONSULTATION", S_TTL)],
    ], colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1),NAVY),
        ("TOPPADDING",    (0,0),(-1,-1),12),
        ("BOTTOMPADDING", (0,0),(-1,-1),12),
        ("LEFTPADDING",   (0,0),(-1,-1),10),
        ("RIGHTPADDING",  (0,0),(-1,-1),10),
    ]))
    story += [banner, Spacer(1, 4*mm)]

    # ── 1. Résumé DPI (bullet format from cr.dpi_textuel) ────────────────────
    if dpi_textuel:
        story += [make_table([sec("Résumé du dossier patient"), row("", dpi_textuel)]),
                  Spacer(1, 3*mm)]

    # ── 2. Motif ──────────────────────────────────────────────────────────────
    if motif:
        story += [make_table([sec("Motif de consultation"), row("Motif", motif)]),
                  Spacer(1, 3*mm)]

    # ── 3. Antécédents & Allergies ────────────────────────────────────────────
    antec_med  = antecedents.get("medicaux") or []
    antec_chir = antecedents.get("chirurgicaux") or []
    antec_fam  = antecedents.get("familiaux") or []
    antec_gyn  = antecedents.get("gynecologiques") or []
    if any([antec_med, antec_chir, antec_fam, antec_gyn, allergies]):
        entries = [sec("Antécédents & Allergies")]; shade = False
        for a in antec_med:
            entries.append(row("Médical", a, shade)); shade = not shade
        for a in antec_chir:
            entries.append(row("Chirurgical", a, shade)); shade = not shade
        for a in antec_fam:
            entries.append(row("Familial", a, shade)); shade = not shade
        for a in antec_gyn:
            entries.append(row("Gynécologique", a, shade)); shade = not shade
        if allergies:
            entries.append(row("Allergies", ", ".join(allergies), shade))
        else:
            entries.append(row("Allergies", "Aucune connue", shade))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 4. Mode de vie ────────────────────────────────────────────────────────
    mv_items = {k: v for k, v in mode_vie.items() if v}
    if mv_items:
        entries = [sec("Mode de vie")]; shade = False
        labels  = {"tabac": "Tabac", "alcool": "Alcool", "drogues": "Drogues",
                   "activite_physique": "Activité physique",
                   "voyages_recents": "Voyages récents", "autre": "Autre"}
        for k, v in mv_items.items():
            entries.append(row(labels.get(k, k.capitalize()), v, shade)); shade = not shade
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 5. Traitements habituels ──────────────────────────────────────────────
    if trts_hab:
        entries = [sec("Traitements habituels")]; shade = False
        for t in trts_hab:
            nom_t = t.get("nom_commercial") or t.get("molecule") or "—"
            pos   = t.get("posologie") or ""
            entries.append(row(nom_t, pos, shade)); shade = not shade
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 6. Interrogatoire ─────────────────────────────────────────────────────
    symp_gen = interrogatoire_d.get("symptomes_generaux") or {}
    symp_org = interrogatoire_d.get("symptomes_par_organe") or []
    autres   = interrogatoire_d.get("autres") or []

    # Only show symptomes_generaux fields that are not null
    gen_items = {}
    if isinstance(symp_gen, dict):
        labels_gen = {"asthenie": "Asthénie", "perte_de_poids": "Perte de poids",
                      "fievre": "Fièvre", "anorexie": "Anorexie"}
        gen_items = {labels_gen[k]: v for k, v in symp_gen.items()
                     if k in labels_gen and v}

    if gen_items or symp_org or autres:
        entries = [sec("Interrogatoire")]
        if gen_items:
            entries.append(row("Symptômes généraux", ", ".join(f"{k}: {v}" for k, v in gen_items.items())))
        for s in symp_org:
            detail = s.get("symptomes", "")
            if s.get("date_debut"):  detail += f" — depuis {s['date_debut']}"
            if s.get("intensite"):   detail += f" — intensité : {s['intensite']}"
            if s.get("evolution"):   detail += f" — {s['evolution']}"
            if s.get("traitements_testes"): detail += f" — traitements testés : {s['traitements_testes']}"
            if s.get("examens_complementaires_realises"):
                detail += f" — examens : {s['examens_complementaires_realises']}"
            entries.append(row(s.get("organe", ""), detail))
        if autres:
            entries.append(row("Autres", " · ".join(autres)))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 7. Examen clinique ────────────────────────────────────────────────────
    constantes = examen_d.get("constantes") or {}
    exam_spec  = examen_d.get("examen_specifique") or ""
    const_vals = {k: v for k, v in constantes.items() if v is not None}
    const_labels = {"poids_kg": "Poids (kg)", "taille_cm": "Taille (cm)", "imc": "IMC",
                    "pression_arterielle": "Pression artérielle",
                    "frequence_cardiaque": "Fréquence cardiaque (bpm)",
                    "temperature": "Température (°C)", "spo2": "SpO2 (%)",
                    "frequence_respiratoire": "Fréquence respiratoire"}
    if const_vals or exam_spec:
        entries = [sec("Examen clinique")]; shade = False
        for k, v in const_vals.items():
            entries.append(row(const_labels.get(k, k), str(v), shade)); shade = not shade
        if exam_spec:
            entries.append(row("Examen spécifique", exam_spec, shade))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 8. Conclusion ─────────────────────────────────────────────────────────
    points_cles  = conclusion.get("points_cles") or []
    diagnostics  = conclusion.get("diagnostic") or []
    prop_ther    = conclusion.get("proposition_therapeutique") or {}
    proch        = conclusion.get("prochaine_consultation") or ""

    medics  = prop_ther.get("medicaments") or [] if isinstance(prop_ther, dict) else []
    paramed = prop_ther.get("paramedical") or [] if isinstance(prop_ther, dict) else []
    excomp  = prop_ther.get("examens_complementaires") or [] if isinstance(prop_ther, dict) else []
    orient  = prop_ther.get("orientation") or "" if isinstance(prop_ther, dict) else ""

    if any([points_cles, diagnostics, medics, paramed, excomp, orient, proch]):
        entries = [sec("Conclusion")]
        if points_cles:
            entries.append(row("Points clés", " · ".join(points_cles)))
        if diagnostics:
            entries.append(row("Diagnostic", " · ".join(diagnostics) if isinstance(diagnostics, list) else str(diagnostics)))
        if medics:
            entries.append(row("Médicaments", " · ".join(medics)))
        if paramed:
            entries.append(row("Paramédical", " · ".join(paramed)))
        if excomp:
            entries.append(row("Examens complémentaires", " · ".join(excomp)))
        if orient:
            entries.append(row("Orientation", orient))
        if proch:
            entries.append(row("Prochaine consultation", proch))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 9. Ordonnance ─────────────────────────────────────────────────────────
    if prescs:
        entries = [sec("Ordonnance")]
        for i, line in enumerate(prescs):
            entries.append(row(f"Prescription {i+1}", line, i % 2 == 1))
        story += [make_table(entries), Spacer(1, 3*mm)]

    story += [
        Spacer(1, 4*mm),
        HRFlowable(width=W, thickness=0.5, color=BORDER),
        Spacer(1, 2*mm),
        Paragraph("Document généré par Napoleon · Scaleway GPU · Confidentiel — à valider par le médecin", S_FOOT),
    ]

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(audio_bytes: bytes, audio_filename: str):
    total_pipeline_start = time.perf_counter()
    total_llm_time = 0
    from prompts import build_dpi_prompt, build_cr_prompt, build_review_prompt, build_diarization_prompt

    STEPS = [
        "Transcription audio",
        "Correction transcription",
        "Vérification anti-hallucination",
        "Relecture médicale",
        "Diarisation (identification des locuteurs)",
        "Enrichissement DPI",
        "Rédaction du compte-rendu",
        "Génération du PDF",
    ]
    n = len(STEPS)

    progress_bar = st.progress(0.0)
    status_text  = st.empty()
    step_display = st.empty()

    def update(i):
        progress_bar.progress(i / n)
        status_text.markdown(f"**Étape {i}/{n} — {STEPS[i-1]}…**")
        lines = []
        for j, s in enumerate(STEPS):
            if j < i - 1:
                lines.append(f'<div class="pipeline-step"><span class="done">✓</span> {s}</div>')
            elif j == i - 1:
                lines.append(f'<div class="pipeline-step"><span class="spin">▶</span> {s}</div>')
            else:
                lines.append(f'<div class="pipeline-step"><span class="wait">○</span> {s}</div>')
        step_display.markdown("".join(lines), unsafe_allow_html=True)

    def finish():
        progress_bar.progress(1.0)
        status_text.markdown("**✅ Pipeline terminé !**")
        lines = [f'<div class="pipeline-step"><span class="done">✓</span> {s}</div>' for s in STEPS]
        step_display.markdown("".join(lines), unsafe_allow_html=True)

    # Step 1 — Transcription
    update(1)

    stt_start = time.perf_counter()

    try:
        text, wc = transcribe_audio(audio_bytes, audio_filename)
    except Exception as e:
        st.error(f"❌ Transcription échouée : {e}")
        return
    st.session_state.stt_time = (time.perf_counter() - stt_start)
    st.session_state.transcript     = text
    st.session_state.word_count     = wc
    st.session_state.audio_filename = audio_filename

    # Step 2 — LLM error correction
    update(2)
    llm_start = time.perf_counter()

    from prompts import build_error_correction_prompt

    correction = call_llm(build_error_correction_prompt(text), max_tokens=2000)

    total_llm_time += (time.perf_counter() - llm_start)

    if "error" in correction:
        st.warning("⚠️ Correction impossible — utilisation de la transcription brute.")
        corrected_text = text
    else:
        corrected_text = correction.get("transcript_corrigee", text)

    st.session_state.raw_transcript = text
    st.session_state.corrected_transcript = corrected_text
    text = corrected_text

    # Step 3 — anti-hallucination
    update(3)
    is_hal, reason = detect_hallucination(text)
    st.session_state.hallucination_ok = not is_hal
    st.session_state.hallucination_reason = reason
    llm_start = time.perf_counter()

    review_result = call_llm(build_review_prompt(text), max_tokens=2000)

    review_time = time.perf_counter() - llm_start
    total_llm_time += review_time
    st.session_state.review = review_result if "error" not in review_result else None

    # Step 4 - Diarization
    update(4)
    # max_tokens=3000: Scaleway llama-3.3-70b caps at ~4096 output tokens.
    # 8000 was causing silent failures.
    diarization_result = call_llm(build_diarization_prompt(text), max_tokens=3000)
    if "error" not in diarization_result:
        segments = diarization_result.get("segments", [])
        # Build labeled_transcript in Python — no risk of JSON truncation
        lines = []
        labeled_text_parts = []
        for seg in segments:
            speaker = seg.get("speaker", "?")
            seg_text = seg.get("text", "")
            lines.append({"speaker": speaker, "text": seg_text})
            labeled_text_parts.append(f"{speaker}: {seg_text}")
        labeled = "\n".join(labeled_text_parts)
        st.session_state.diarization = {"segments": lines, "labeled_transcript": labeled}
        if labeled:
            text = labeled
    else:
        st.warning(f"⚠️ Diarisation échouée (pipeline continué sans) : {diarization_result.get('error', 'erreur inconnue')}")
        st.session_state.diarization = None

    # Step 5 — DPI enrichment
    update(5)
    llm_start = time.perf_counter()

    dpi_result = call_llm(build_dpi_prompt(text, None), max_tokens=4000)

    dpi_time = time.perf_counter() - llm_start
    total_llm_time += dpi_time
    if "error" in dpi_result:
        st.error(f"❌ Erreur DPI : {dpi_result['error']}")
        return
    st.session_state.enriched_dpi = dpi_result

    # Step 6 — CR generation
    update(6)
    llm_start = time.perf_counter()

    cr_result = call_llm(
        build_cr_prompt(text, dpi_result),
        max_tokens=3000
    )

    cr_time = time.perf_counter() - llm_start
    total_llm_time += cr_time
    if "error" in cr_result:
        st.error(f"❌ Erreur CR : {cr_result['error']}")
        return
    st.session_state.cr = cr_result

    # Step 7 — PDF
    update(7)
    stem = Path(audio_filename).stem
    st.session_state.pdf_bytes     = generate_pdf(dpi_result, cr_result, stem)
    st.session_state.llm_time = total_llm_time

    st.session_state.total_time = (
        time.perf_counter() - total_pipeline_start
    )
    st.session_state.pipeline_done = True

    finish()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_launch, tab_transcription, tab_review, tab_extraction, tab_pdf = st.tabs([
    "🚀  Lancer",
    "🎙️  Transcription",
    "🔍  Relecture",
    "🧠  Extraction",
    "📋  Rapport PDF",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — LAUNCH
# ════════════════════════════════════════════════════════════════════════════
with tab_launch:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Charger l'audio")
        input_mode = st.radio(
            "Source audio",
            ["🎙️ Enregistrer au micro", "📁 Charger un fichier"],
            label_visibility="collapsed",
            horizontal=True,
        )

        uploaded = None
        if input_mode.startswith("🎙️"):
            st.caption("Cliquez pour démarrer l'enregistrement, puis à nouveau pour l'arrêter.")
            mic_audio = st.audio_input("Enregistrement micro", label_visibility="collapsed")
            if mic_audio:
                uploaded = mic_audio
                uploaded.name = "enregistrement_micro.wav"  # st.audio_input doesn't set .name by default
        else:
            st.caption("Formats : .m4a, .wav, .mp3, .flac, .ogg")
            uploaded = st.file_uploader("Audio", type=["m4a","wav","mp3","flac","ogg"], label_visibility="collapsed")

        if uploaded:
            st.audio(uploaded)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            st.caption("Kyutai STT 1B (en/fr) · llama-3.3-70b · Scaleway")
            launch = st.button("Lancer le pipeline complet", use_container_width=True)

            if launch:
                run_pipeline(uploaded.read(), uploaded.name)

    with col2:
        if st.session_state.pipeline_done:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Résultats")
            stem = Path(st.session_state.audio_filename or "consultation").stem

            wc        = st.session_state.word_count
            hal_label = "✅ OK" if st.session_state.hallucination_ok else "⚠️ Détectée"
            rev_count = len((st.session_state.review or {}).get("corrections", []))
            st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box"><div class="value">{wc}</div><div class="label">Mots</div></div>
                    <div class="metric-box"><div class="value">{st.session_state.stt_time:.1f}s</div><div class="label">STT</div></div>
                    <div class="metric-box"><div class="value">{st.session_state.llm_time:.1f}s</div><div class="label">LLM</div></div>
                    <div class="metric-box"><div class="value">{st.session_state.total_time:.1f}s</div><div class="label">Total</div></div>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><div class="value">{hal_label}</div><div class="label">Hallucination</div></div>
                    <div class="metric-box"><div class="value">{rev_count}</div><div class="label">Corrections</div></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**Téléchargements :**")
            if st.session_state.enriched_dpi:
                st.download_button("⬇  DPI enrichi (JSON)",
                    key="enriched_dpi_download_launch",
                    data=json.dumps(st.session_state.enriched_dpi, ensure_ascii=False, indent=2),
                    file_name=f"dpi_{stem}.json", mime="application/json", use_container_width=True)
            if st.session_state.cr:
                st.download_button("⬇  Compte-rendu (JSON)",
                    key="download_cr",
                    data=json.dumps(st.session_state.cr, ensure_ascii=False, indent=2),
                    file_name=f"cr_{stem}.json", mime="application/json", use_container_width=True)
            if st.session_state.pdf_bytes:
                st.download_button("⬇  Rapport PDF",
                                   key="download_pdf",
                    data=st.session_state.pdf_bytes,
                    file_name=f"rapport_{stem}.pdf",
                    mime="application/pdf", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:#94A3B8;text-align:center">
                <div style="font-size:3rem"></div>
                <p>Chargez un fichier audio et cliquez sur <b>Lancer</b><br>pour démarrer le pipeline automatique.</p>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transcription (viewer)
# ════════════════════════════════════════════════════════════════════════════
with tab_transcription:
    if not st.session_state.transcript:
        st.info("Lancez le pipeline depuis l'onglet Lancer.")
    else:
        col_left, col_right = st.columns([1, 1], gap="large")
        with col_left:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Informations")
            is_hal = not st.session_state.hallucination_ok
            reason = st.session_state.hallucination_reason or ""
            if is_hal:
                st.markdown(f'<div class="alert-error">⚠️ <b>Hallucination détectée</b><br>{reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-ok">✓ <b>Aucune hallucination</b><br>{reason}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="value">{st.session_state.word_count}</div><div class="label">Mots</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            if st.session_state.diarization:
                st.markdown('<div class="step-card">', unsafe_allow_html=True)
                st.markdown("#### Transcription diarisée")
                st.caption("Locuteurs identifiés par l'IA — Médecin vs Patient")
                labeled = st.session_state.diarization.get("labeled_transcript", "")
                for line in labeled.split("\n"):
                    if line.startswith("Médecin:"):
                        st.markdown(f'<p style="color:#0D1B3E;margin:2px 0"><b>{line}</b></p>', unsafe_allow_html=True)
                    elif line.startswith("Patient:"):
                        st.markdown(f'<p style="color:#028090;margin:2px 0">{line}</p>', unsafe_allow_html=True)
                    elif line.strip():
                        st.markdown(f'<p style="color:#64748B;margin:2px 0">{line}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Transcription brute")
            st.caption("Modifiez si nécessaire — relancez le pipeline pour ré-extraire.")
            edited = st.text_area("Transcription :", value=st.session_state.transcript,
                                  height=300, label_visibility="collapsed")
            if edited != st.session_state.transcript:
                st.session_state.transcript = edited
                st.caption("✏️ Modifiée manuellement — relancez le pipeline pour appliquer.")
            st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Medical review (viewer)
# ════════════════════════════════════════════════════════════════════════════
with tab_review:
    if not st.session_state.review:
        st.info("Lancez le pipeline depuis l'onglet Lancer.")
    else:
        review = st.session_state.review
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Résumé de la relecture")
        st.markdown(review.get("resume", "—"))
        st.markdown("</div>", unsafe_allow_html=True)

        corrections = review.get("corrections", [])
        alertes     = review.get("alertes", [])

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown(f"#### Corrections proposées ({len(corrections)})")
            if corrections:
                for c in corrections:
                    conf_color = {"haute": "#059669", "moyenne": "#D97706", "faible": "#DC2626"}.get(
                        c.get("confiance", ""), "#64748B")
                    st.markdown(f"""
                    <div style="border-left:3px solid {conf_color};padding:0.5rem 0.8rem;margin-bottom:0.5rem;background:#F8FAFC;border-radius:0 6px 6px 0">
                        <b>{c.get('original','')}</b> → <b style="color:{conf_color}">{c.get('corrige','')}</b>
                        <span style="background:#E2E8F0;border-radius:4px;padding:0.1rem 0.4rem;font-size:0.75rem;margin-left:0.4rem">{c.get('type','')}</span>
                        <div style="font-size:0.8rem;color:#64748B;margin-top:0.3rem">{c.get('explication','')}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-ok">✓ Aucune correction nécessaire</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown(f"#### Alertes ({len(alertes)})")
            if alertes:
                for a in alertes:
                    st.markdown(f"""
                    <div class="alert-warn" style="margin-bottom:0.5rem">
                        <b>Passage :</b> {a.get('texte','')}<br>
                        <span style="font-size:0.85rem">{a.get('raison','')}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-ok">✓ Aucune alerte</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if review.get("transcription_corrigee"):
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Transcription corrigée")
            st.markdown(review["transcription_corrigee"])
            st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Extraction (viewer)
# ════════════════════════════════════════════════════════════════════════════
with tab_extraction:
    if not st.session_state.enriched_dpi and not st.session_state.cr:
        st.info("Lancez le pipeline depuis l'onglet Lancer.")
    else:
        col_left, col_right = st.columns([1, 2], gap="large")
        with col_left:
            stem = Path(st.session_state.audio_filename or "consultation").stem
            if st.session_state.enriched_dpi:
                st.download_button("⬇  DPI enrichi (JSON)",
                    key="download_dpi_launch",
                    data=json.dumps(st.session_state.enriched_dpi, ensure_ascii=False, indent=2),
                    file_name=f"dpi_{stem}.json", mime="application/json", use_container_width=True)
            if st.session_state.cr:
                st.download_button("⬇  Compte-rendu (JSON)",
                    key="download_cr_launch",
                    data=json.dumps(st.session_state.cr, ensure_ascii=False, indent=2),
                    file_name=f"cr_{stem}.json", mime="application/json", use_container_width=True)
        with col_right:
            if st.session_state.enriched_dpi:
                st.markdown('<div class="json-section"><div class="json-section-header">🗂️ DPI enrichi</div></div>', unsafe_allow_html=True)
                with st.expander("Voir le JSON", expanded=False):
                    st.json(st.session_state.enriched_dpi)
            if st.session_state.cr:
                st.markdown('<div class="json-section"><div class="json-section-header">📝 Compte-rendu & Prescriptions</div></div>', unsafe_allow_html=True)
                cr = st.session_state.cr
                if cr.get("cr_textuel"):
                    st.markdown(cr["cr_textuel"])
                if cr.get("prescription_lines"):
                    st.markdown("**Ordonnance :**")
                    for line in cr["prescription_lines"]:
                        st.markdown(f"- {line}")
                if cr.get("dpi_textuel"):
                    with st.expander("Résumé DPI textuel"):
                        st.markdown(cr["dpi_textuel"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PDF (viewer)
# ════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    if not st.session_state.pdf_bytes:
        st.info("Lancez le pipeline depuis l'onglet Lancer.")
    else:
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Rapport PDF")
            st.caption("Compte-rendu structuré — à valider par le médecin")
            stem = Path(st.session_state.audio_filename or "consultation").stem
            st.download_button("⬇  Télécharger le PDF",
                                key="download_pdf_launch",
                data=st.session_state.pdf_bytes,
                file_name=f"rapport_{stem}.pdf",
                mime="application/pdf", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Aperçu")
            cr = st.session_state.cr or {}
            if cr.get("cr_textuel"):
                st.markdown(cr["cr_textuel"])
            if cr.get("prescription_lines"):
                st.markdown("---")
                st.markdown("**Ordonnance :**")
                for line in cr["prescription_lines"]:
                    st.markdown(f"- {line}")
            st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #E8E6E0;margin-top:3rem">
<p style="text-align:center;color:#94A3B8;font-size:0.8rem">
Napoleon · Pipeline médical IA · Scaleway GPU · Confidentiel
</p>
""", unsafe_allow_html=True)
