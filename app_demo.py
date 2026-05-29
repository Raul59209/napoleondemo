"""
app_demo.py — Napoleon Medical Pipeline
========================================
Flow:
  Tab 1 : Upload audio → transcribe (Scaleway Whisper) → hallucination check
  Tab 2 : LLM extraction → enriched DPI + CR textuels (two Scaleway calls)
  Tab 3 : PDF + downloads
"""

import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    "transcript":       None,
    "hallucination_ok": None,
    "audio_filename":   None,
    "existing_dpi":     None,  # uploaded by user (optional)
    "enriched_dpi":     None,  # output of Call 1
    "cr":               None,  # output of Call 2 (cr_modele.json)
    "pdf_bytes":        None,
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
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = scw_client.audio.transcriptions.create(
                model="whisper-large-v3", file=f, language="fr",
            )
        text = result.text.strip()
        return text, len(text.split())
    finally:
        os.unlink(tmp_path)


def call_llm(prompt: str, max_tokens: int = 4000) -> dict:
    try:
        response = scw_client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"JSON invalide : {e}", "raw": raw[:500]}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATOR
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

    # ── Pull data ─────────────────────────────────────────────────────────────
    dpi          = enriched_dpi.get("dpi") or {}
    admin        = dpi.get("administratif") or {}
    dossier      = dpi.get("dossier_medical") or {}
    docs         = dpi.get("documents") or {}
    historique   = dossier.get("historique_medical") or {}
    traitements  = dossier.get("traitements") or {}
    mode_vie     = dossier.get("mode_de_vie") or {}
    consultations = ls(docs, "consultations")
    last_c       = consultations[-1] if consultations else {}
    etat_civil   = admin.get("etat_civil") or {}
    identite     = admin.get("identite_usage") or {}
    dpi_textuel  = (cr or {}).get("dpi_textuel") or ""
    cr_textuel   = (cr or {}).get("cr_textuel") or ""
    prescs       = (cr or {}).get("prescription_lines") or []

    story = []

    # ── Banner ────────────────────────────────────────────────────────────────
    nom    = sv(identite,"nom_utilise") if sv(identite,"nom_utilise") != "—" else sv(etat_civil,"nom_naissance")
    prenom = sv(identite,"prenom_utilise", default="")
    ddn    = sv(etat_civil,"date_naissance", default="")
    date_c = sv(last_c,"date", default="")
    sub_parts = [f"{prenom} {nom}".strip()]
    if ddn    != "—": sub_parts.append(f"Né(e) le {ddn}")
    if date_c != "—": sub_parts.append(f"Consultation du {date_c}")

    banner = Table([
        [Paragraph("🩺  NAPOLEON — COMPTE-RENDU DE CONSULTATION", S_TTL)],
        [Paragraph("  ·  ".join(sub_parts), S_SUB)],
    ], colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1),NAVY),
        ("TOPPADDING",    (0,0),(-1,-1),8),
        ("BOTTOMPADDING", (0,0),(-1,-1),8),
        ("LEFTPADDING",   (0,0),(-1,-1),10),
        ("RIGHTPADDING",  (0,0),(-1,-1),10),
    ]))
    story += [banner, Spacer(1, 4*mm)]

    # ── 1. Résumé DPI ─────────────────────────────────────────────────────────
    if dpi_textuel:
        story += [make_table([sec("Résumé du dossier patient"), row("DPI", dpi_textuel)]),
                  Spacer(1, 3*mm)]

    # ── 2. Motif ──────────────────────────────────────────────────────────────
    motif = sv(last_c, "motif_de_consultation")
    if motif != "—":
        story += [make_table([sec("Motif de consultation"), row("Motif", motif)]),
                  Spacer(1, 3*mm)]

    # ── 3. Antécédents ────────────────────────────────────────────────────────
    path_chron = ls(historique, "pathologies_chroniques")
    antec_med  = ls(historique, "antecedents_medicaux")
    antec_chir = ls(historique, "antecedents_chirurgicaux")
    antec_fam  = ls(historique, "familiaux")
    allergies  = ls(historique, "allergies")
    gyneco     = historique.get("gynecologique") or {}

    if any([path_chron, antec_med, antec_chir, antec_fam, allergies,
            gyneco.get("gestite"), gyneco.get("contraception_actuelle")]):
        entries = [sec("Historique médical & Antécédents")]
        shade = False
        for p in path_chron:
            cim = sv(p,"code_cim10",default=""); ald = " · ALD" if p.get("ald") else ""
            entries.append(row("Pathologie chronique", f"{sv(p,'libelle')}{' ('+cim+')' if cim!='—' else ''}{ald}", shade)); shade = not shade
        for a in antec_med:
            entries.append(row("Antécédent médical", sv(a,"libelle"), shade)); shade = not shade
        for a in antec_chir:
            etab = sv(a,"etablissement",default="")
            entries.append(row("Antécédent chirurgical", f"{sv(a,'libelle')}{' ('+etab+')' if etab!='—' else ''}", shade)); shade = not shade
        for a in antec_fam:
            lien = sv(a,"lien_parente",default="")
            entries.append(row("Antécédent familial", f"{lien+' : ' if lien!='—' else ''}{sv(a,'libelle')}", shade)); shade = not shade
        for a in allergies:
            manif = sv(a,"manifestation",default="")
            entries.append(row("Allergie", f"{sv(a,'substance')}{' → '+manif if manif!='—' else ''}", shade)); shade = not shade
        if gyneco.get("gestite") is not None or gyneco.get("contraception_actuelle"):
            parts = []
            g  = sv(gyneco,"gestite",default=""); p2 = sv(gyneco,"parite",default="")
            c  = sv(gyneco,"contraception_actuelle",default="")
            if g  != "—": parts.append(f"G{g}")
            if p2 != "—": parts.append(f"P{p2}")
            if c  != "—": parts.append(f"Contraception : {c}")
            if parts: entries.append(row("Gynécologique", " · ".join(parts), shade))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 4. Mode de vie ────────────────────────────────────────────────────────
    tabac    = mode_vie.get("tabac") or {}
    alcool   = mode_vie.get("alcool") or {}
    activite = ls(mode_vie, "activite_physique")
    if any([tabac.get("quantite_par_frequence"), alcool.get("quantite_par_frequence"), activite]):
        entries = [sec("Mode de vie")]; shade = False
        if tabac.get("quantite_par_frequence"):
            pa = sv(tabac,"paquets_annees",default="")
            entries.append(row("Tabac", f"{sv(tabac,'quantite_par_frequence')}{' · '+pa+' PA' if pa!='—' else ''}", shade)); shade = not shade
        if alcool.get("quantite_par_frequence"):
            entries.append(row("Alcool", sv(alcool,"quantite_par_frequence"), shade)); shade = not shade
        for a in activite:
            entries.append(row("Activité physique", f"{sv(a,'type')} — {sv(a,'frequence',default='')}".strip(" —"), shade)); shade = not shade
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 5. Traitements ────────────────────────────────────────────────────────
    trts_hab  = ls(traitements, "habituels")
    trts_ponc = ls(traitements, "ponctuels")
    if trts_hab or trts_ponc:
        entries = [sec("Traitements")]; shade = False
        for t in trts_hab:
            nom_t = sv(t,"nom_commercial") if sv(t,"nom_commercial") != "—" else sv(t,"molecule")
            pos   = sv(t,"posologie",default=""); indic = sv(t,"indication",default="")
            entries.append(row(f"{nom_t} (habituel)", f"{pos}{' · '+indic if indic!='—' else ''}", shade)); shade = not shade
        for t in trts_ponc:
            nom_t  = sv(t,"nom_commercial") if sv(t,"nom_commercial") != "—" else sv(t,"molecule")
            pos    = sv(t,"posologie",default="")
            df     = sv(t,"date_fin",default="")
            df_str = f" · jusqu'au {df}" if df != "—" else ""
            entries.append(row(f"{nom_t} (ponctuel)", f"{pos}{df_str}", shade)); shade = not shade
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── 6. Compte-rendu ───────────────────────────────────────────────────────
    interrogatoire = sv(last_c, "interrogatoire")
    examen         = sv(last_c, "examen_clinique")
    conclusion     = sv(last_c, "conclusion")
    if any(v != "—" for v in [interrogatoire, examen, conclusion]):
        entries = [sec("Compte-rendu de la consultation")]
        if interrogatoire != "—": entries.append(row("Interrogatoire", interrogatoire))
        if examen         != "—": entries.append(row("Examen clinique", examen))
        if conclusion     != "—": entries.append(row("Conclusion",      conclusion))
        story += [make_table(entries), Spacer(1, 3*mm)]
    elif cr_textuel:
        story += [make_table([sec("Compte-rendu de la consultation"),
                               row("Compte-rendu", cr_textuel)]),
                  Spacer(1, 3*mm)]

    # ── 7. Ordonnance ─────────────────────────────────────────────────────────
    if prescs:
        entries = [sec("Ordonnance")]
        for i, line in enumerate(prescs):
            entries.append(row(f"Prescription {i+1}", line, i % 2 == 1))
        story += [make_table(entries), Spacer(1, 3*mm)]

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 4*mm),
        HRFlowable(width=W, thickness=0.5, color=BORDER),
        Spacer(1, 2*mm),
        Paragraph("Document généré par Napoleon · Scaleway GPU · Confidentiel — à valider par le médecin", S_FOOT),
    ]

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎙️  Transcription", "🧠  Extraction", "📋  Rapport PDF"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transcription
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### 1. DPI existant (optionnel)")
        st.caption("Chargez le DPI JSON du patient si disponible. Sinon il sera construit depuis l'audio.")
        dpi_file = st.file_uploader("DPI JSON", type=["json"], label_visibility="collapsed")
        if dpi_file:
            try:
                st.session_state.existing_dpi = json.load(dpi_file)
                st.markdown('<div class="alert-ok">✓ DPI chargé</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"JSON invalide : {e}")
        elif st.session_state.existing_dpi:
            st.markdown('<div class="alert-ok">✓ DPI déjà chargé en session</div>', unsafe_allow_html=True)
        else:
            st.caption("Aucun DPI — il sera construit à partir de l'audio.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Charger l'audio")
        st.caption("Formats : .m4a, .wav, .mp3, .flac, .ogg")
        uploaded = st.file_uploader("Audio", type=["m4a","wav","mp3","flac","ogg"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            st.audio(uploaded)
            st.session_state.audio_filename = uploaded.name
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### 3. Transcrire")
            st.caption("Whisper large-v3 · Scaleway GPU · Français")
            if st.button("▶  Lancer la transcription", use_container_width=True):
                with st.spinner("Transcription en cours..."):
                    try:
                        text, wc = transcribe_audio(uploaded.read(), uploaded.name)
                        st.session_state.transcript = text
                        st.session_state.hallucination_ok = None
                        st.success("Transcription terminée ✓")
                        st.markdown(f"""
                        <div class="metric-row">
                            <div class="metric-box"><div class="value">🟢</div><div class="label">Scaleway GPU</div></div>
                            <div class="metric-box"><div class="value">{wc}</div><div class="label">Mots</div></div>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.session_state.transcript:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### 4. Vérification anti-hallucination")
            is_hal, reason = detect_hallucination(st.session_state.transcript)
            st.session_state.hallucination_ok = not is_hal
            if is_hal:
                st.markdown(f'<div class="alert-error">⚠️ <b>Hallucination détectée</b><br>{reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-ok">✓ <b>Aucune hallucination</b><br>{reason}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### 5. Transcription finale")
            st.caption("Modifiez si nécessaire avant l'extraction")
            edited = st.text_area("Transcription :", value=st.session_state.transcript, height=300, label_visibility="collapsed")
            if edited != st.session_state.transcript:
                st.session_state.transcript = edited
                st.caption("✏️ Modifiée manuellement")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:#94A3B8;text-align:center">
                <div style="font-size:3rem">🎙️</div><p>Chargez un fichier audio et lancez la transcription</p>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Extraction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.transcript:
        st.info("Complétez d'abord la transcription (onglet 1).")
    else:
        if st.session_state.hallucination_ok is False:
            st.markdown('<div class="alert-warn">⚠️ Hallucination détectée — vérifiez la transcription avant d\'extraire.</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Extraction LLM")
            dpi_status = "✓ DPI existant chargé" if st.session_state.existing_dpi else "Ø Pas de DPI — construction depuis l'audio"
            st.caption(f"llama-3.3-70b-instruct · Scaleway · {dpi_status}")

            if st.button("🧠  Lancer l'extraction (2 appels)", use_container_width=True):
                from prompts import build_dpi_prompt, build_cr_prompt
                progress = st.progress(0)
                status   = st.empty()

                status.caption("Appel 1/2 — Enrichissement du DPI...")
                dpi_result = call_llm(
                    build_dpi_prompt(st.session_state.transcript, st.session_state.existing_dpi),
                    max_tokens=4000
                )
                progress.progress(0.5)

                if "error" in dpi_result:
                    st.error(f"Erreur DPI : {dpi_result['error']}")
                else:
                    st.session_state.enriched_dpi = dpi_result
                    status.caption("Appel 2/2 — Rédaction du compte-rendu...")
                    cr_result = call_llm(
                        build_cr_prompt(st.session_state.transcript, dpi_result),
                        max_tokens=3000
                    )
                    progress.progress(1.0)
                    if "error" in cr_result:
                        st.error(f"Erreur CR : {cr_result['error']}")
                    else:
                        st.session_state.cr = cr_result
                        st.success("Extraction terminée ✓")

                status.empty()
                progress.empty()

            st.markdown("</div>", unsafe_allow_html=True)

            stem = Path(st.session_state.audio_filename or "consultation").stem
            if st.session_state.enriched_dpi:
                st.download_button("⬇  DPI enrichi (JSON)",
                    data=json.dumps(st.session_state.enriched_dpi, ensure_ascii=False, indent=2),
                    file_name=f"dpi_{stem}.json", mime="application/json", use_container_width=True)
            if st.session_state.cr:
                st.download_button("⬇  Compte-rendu (JSON)",
                    data=json.dumps(st.session_state.cr, ensure_ascii=False, indent=2),
                    file_name=f"cr_{stem}.json", mime="application/json", use_container_width=True)

        with col_right:
            if st.session_state.enriched_dpi or st.session_state.cr:
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
            else:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:#94A3B8;text-align:center">
                    <div style="font-size:3rem">🧠</div><p>Lancez l'extraction pour voir les résultats</p>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PDF
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.enriched_dpi or not st.session_state.cr:
        st.info("Complétez d'abord l'extraction LLM (onglet 2).")
    else:
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Générer le rapport PDF")
            st.caption("Compte-rendu structuré — à valider par le médecin")
            stem = Path(st.session_state.audio_filename or "consultation").stem

            if st.button("📋  Générer le PDF", use_container_width=True):
                with st.spinner("Génération..."):
                    pdf = generate_pdf(st.session_state.enriched_dpi, st.session_state.cr, stem)
                if pdf:
                    st.session_state.pdf_bytes = pdf
                    st.success("PDF généré ✓")
                else:
                    st.error("Erreur — vérifiez que reportlab est installé.")

            if st.session_state.pdf_bytes:
                st.download_button("⬇  Télécharger le PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"rapport_{stem}.pdf",
                    mime="application/pdf", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("#### Aperçu")
            cr = st.session_state.cr
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