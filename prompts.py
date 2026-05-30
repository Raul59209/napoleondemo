"""
prompts.py — Napoleon LLM Prompts
===================================
Two calls per consultation:
  1. build_dpi_prompt  → returns enriched DPI JSON (DPI.json schema)
  2. build_cr_prompt   → returns CR JSON (cr_modele.json schema)
"""

import json


def build_dpi_prompt(transcript: str, existing_dpi: dict | None = None) -> str:
    """
    Call 1: extract consultation data from the transcript into the agreed schema.
    If existing_dpi is provided, enrich it; otherwise build from scratch.
    """

    dpi_schema = {
        "motif_de_consultation": "string ou null",
        "historique_medical": "string — contexte médical évoqué ou null",
        "antecedents": {
            "medicaux": ["liste de strings ou tableau vide"],
            "chirurgicaux": ["liste de strings ou tableau vide"],
            "gynecologiques": ["liste de strings ou tableau vide"],
            "familiaux": ["liste de strings ou tableau vide"]
        },
        "mode_de_vie": {
            "tabac": "string ou null",
            "alcool": "string ou null",
            "drogues": "string ou null",
            "activite_physique": "string ou null",
            "voyages_recents": "string ou null",
            "autre": "string ou null"
        },
        "vaccins": [],
        "traitements_habituels": [
            {
                "nom_commercial": "string",
                "molecule": "string ou null",
                "posologie": "string ou null"
            }
        ],
        "allergies": ["liste de strings ou tableau vide"],
        "interrogatoire": {
            "symptomes_generaux": "string ou null",
            "symptomes_par_organe": [
                {
                    "organe": "string",
                    "symptomes": "string",
                    "date_debut": "string ou null",
                    "evolution": "string ou null",
                    "traitements_testes": "string ou null"
                }
            ],
            "examens_realises": "string ou null"
        },
        "examen_clinique": {
            "constantes": {
                "poids_kg": "number ou null",
                "taille_cm": "number ou null",
                "imc": "number ou null",
                "pression_arterielle": "string ou null",
                "frequence_cardiaque": "number ou null",
                "temperature": "number ou null",
                "spo2": "number ou null"
            },
            "examen_specifique": "string ou null"
        },
        "conclusion": {
            "diagnostic": "string ou null",
            "proposition_therapeutique": "string ou null",
            "examens_complementaires": ["liste de strings ou tableau vide"],
            "orientation": "string ou null",
            "prochaine_consultation": "string ou null"
        }
    }

    if existing_dpi:
        dpi_context = f"""
Le DPI existant du patient est fourni ci-dessous. Enrichis-le avec les informations
extraites de la transcription. Conserve toutes les données existantes et ajoute ou
mets à jour uniquement ce qui est mentionné dans la transcription.

DPI EXISTANT :
{json.dumps(existing_dpi, ensure_ascii=False, indent=2)}
"""
    else:
        dpi_context = """
Aucun DPI existant. Construis le DPI complet à partir des informations extraites
de la transcription. Les champs non mentionnés doivent être null ou tableau vide.
"""

    return f"""Tu es un assistant médical expert en extraction d'informations cliniques.
{dpi_context}

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

INSTRUCTIONS :
- Extrais toutes les informations médicales présentes dans la transcription.
- Ne invente rien : les champs non mentionnés dans la transcription doivent être null ou [].
- Pour les constantes (poids, tension, etc.), n'inscris que ce qui est explicitement dit.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RESPECTER :
{json.dumps(dpi_schema, ensure_ascii=False, indent=2)}
"""


def build_review_prompt(transcript: str) -> str:
    """
    Medical review call: checks the transcript for transcription errors,
    mis-heard medication names, wrong dosages, and anatomical term mistakes.
    Returns a structured JSON with corrections and alerts.
    """

    review_schema = {
        "resume": "string — une phrase résumant l'état général de la transcription",
        "corrections": [
            {
                "original": "string — le mot ou groupe de mots tel que transcrit",
                "corrige": "string — la correction proposée",
                "type": "string — medicament | dosage | anatomie | autre",
                "confiance": "string — haute | moyenne | faible",
                "explication": "string — pourquoi cette correction est proposée"
            }
        ],
        "alertes": [
            {
                "texte": "string — le passage concerné",
                "raison": "string — pourquoi ce passage mérite attention du médecin"
            }
        ],
        "transcription_corrigee": "string — la transcription complète avec les corrections appliquées"
    }

    return f"""Tu es un assistant médical expert en relecture de transcriptions de consultations médicales.

TRANSCRIPTION À VÉRIFIER :
{transcript}

INSTRUCTIONS :
- Vérifie les noms de médicaments : sont-ils orthographiés correctement ?
  Un nom mal transcrit peut indiquer un médicament différent (ex: "Doliprane" vs "Dolipranne").
- Vérifie les dosages : sont-ils cohérents et plausibles pour les médicaments mentionnés ?
- Vérifie les termes anatomiques et médicaux : sont-ils correctement transcrits ?
- Signale dans "alertes" tout passage ambigu que le médecin devrait relire, même si tu n'es
  pas certain qu'il y ait une erreur.
- Si la transcription semble correcte, retourne corrections=[] et alertes=[].
- Ne corrige PAS le style ou la grammaire — uniquement les erreurs médicales potentielles.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(review_schema, ensure_ascii=False, indent=2)}
"""


def build_cr_prompt(transcript: str, enriched_dpi: dict) -> str:
    """
    Call 2: generate CR textuels and prescription lines.
    Receives the transcript + the already-enriched DPI for context.
    Returns cr_modele.json format.
    """

    cr_schema = {
        "dpi_textuel": (
            "string — résumé textuel structuré du DPI complet du patient "
            "(antécédents, traitements habituels, allergies, mode de vie). "
            "Null si le DPI ne contient pas d'informations pertinentes."
        ),
        "cr_textuel": (
            "string — compte-rendu textuel complet de cette consultation : "
            "motif, interrogatoire, examen clinique, diagnostic et proposition "
            "thérapeutique. Rédigé en français médical comme un vrai CR."
        ),
        "prescription_lines": [
            "string — une ligne par médicament prescrit lors de cette consultation. "
            "Format : 'NomCommercial (DCI) — posologie — fréquence — durée'. "
            "Liste vide [] si aucune prescription."
        ]
    }

    return f"""Tu es un assistant médical expert en rédaction de comptes-rendus cliniques.

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

DPI ENRICHI DU PATIENT (pour contexte) :
{json.dumps(enriched_dpi, ensure_ascii=False, indent=2)}

INSTRUCTIONS :
- dpi_textuel : résumé structuré du DPI du patient en français médical clair.
  Inclure antécédents, pathologies chroniques, traitements habituels, allergies,
  mode de vie. Null si le DPI est vide.
- cr_textuel : compte-rendu complet de cette consultation en français médical.
  Inclure motif, interrogatoire, examen clinique, diagnostic, proposition
  thérapeutique. Rédige comme un médecin dans un dossier patient.
- prescription_lines : une entrée par médicament prescrit lors de cette consultation.
  Format : "NomCommercial (DCI) — posologie — fréquence — durée".
  Liste vide [] si aucune prescription.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(cr_schema, ensure_ascii=False, indent=2)}
"""


def build_diarization_prompt(transcript: str) -> str:
    """
    LLM-based speaker diarization for a 2-speaker medical consultation.
    Only returns segments list — labeled_transcript is built in Python.
    """

    schema = {
        "segments": [
            {
                "speaker": "Medecin | Patient",
                "text": "the exact sentence or phrase as it appeared in the transcript"
            }
        ]
    }

    return f"""Tu es un assistant medical expert en analyse de consultations medicales.

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

INSTRUCTIONS :
- Il y a exactement 2 interlocuteurs : le Medecin et le Patient.
- Decoupe en repliques et attribue chaque replique au bon interlocuteur.
- Le Medecin : pose des questions, prescrit, diagnostique, vocabulaire medical.
- Le Patient : decrit symptomes, repond, dit oui/non/d accord.
- Ne modifie PAS le texte des repliques, copie-les exactement.
- Reponds UNIQUEMENT avec du JSON valide, sans texte supplementaire.

SCHEMA :
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""