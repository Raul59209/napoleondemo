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
    Call 1: enrich the DPI with data extracted from the transcript.
    If existing_dpi is None, build from scratch.
    The four consultation fields (motif_de_consultation, interrogatoire,
    examen_clinique, conclusion) go into dpi.documents.consultations.
    """

    dpi_schema = {
        "dpi": {
            "administratif": {
                "identite_usage": {
                    "civilite": "string ou null",
                    "nom_utilise": "string ou null",
                    "prenom_utilise": "string ou null",
                    "coordonnees": {
                        "adresse": {
                            "numero_voie": "string ou null",
                            "type_voie": "string (rue|avenue|boulevard|impasse|chemin|allee|place|route|autre) ou null",
                            "nom_voie": "string ou null",
                            "code_postal": "string ou null",
                            "ville": "string ou null",
                            "pays": "string ou null"
                        },
                        "telephone_mobile": "string ou null",
                        "email": "string ou null"
                    }
                },
                "etat_civil": {
                    "nom_naissance": "string ou null",
                    "prenoms_naissance": ["string"],
                    "date_naissance": "string ISO 8601 ou null",
                    "sexe": "M | F | null"
                },
                "identifiants_couverture": {
                    "medecin_traitant": {
                        "nom": "string ou null",
                        "prenom": "string ou null",
                        "specialite": "string ou null"
                    },
                    "mutuelle": {
                        "nom": "string ou null",
                        "tiers_payant": "boolean ou null"
                    }
                }
            },
            "dossier_medical": {
                "historique_medical": {
                    "pathologies_chroniques": [{
                        "libelle": "string ou null",
                        "code_cim10": "string ou null",
                        "ald": "boolean ou null",
                        "traitements_actuels": [{"nom_commercial": "string ou null", "molecule": "string ou null"}],
                        "commentaire": "string ou null"
                    }],
                    "antecedents_medicaux": [{
                        "libelle": "string ou null",
                        "code_cim10": "string ou null",
                        "date": "string ISO 8601 ou null",
                        "commentaire": "string ou null"
                    }],
                    "antecedents_chirurgicaux": [{
                        "libelle": "string ou null",
                        "date": "string ISO 8601 ou null",
                        "etablissement": "string ou null",
                        "commentaire": "string ou null"
                    }],
                    "familiaux": [{
                        "lien_parente": "string ou null",
                        "libelle": "string ou null",
                        "commentaire": "string ou null"
                    }],
                    "allergies": [{
                        "type": "medicamenteuse | alimentaire | environnementale | autre",
                        "substance": "string ou null",
                        "manifestation": "string ou null"
                    }],
                    "gynecologique": {
                        "contraception_actuelle": "string ou null",
                        "gestite": "integer ou null",
                        "parite": "integer ou null"
                    }
                },
                "traitements": {
                    "habituels": [{
                        "nom_commercial": "string ou null",
                        "molecule": "string ou null",
                        "posologie": "string ou null",
                        "indication": "string ou null"
                    }],
                    "ponctuels": [{
                        "nom_commercial": "string ou null",
                        "molecule": "string ou null",
                        "posologie": "string ou null",
                        "date_fin": "string ISO 8601 ou null"
                    }]
                },
                "mode_de_vie": {
                    "tabac": {
                        "quantite_par_frequence": "string ou null",
                        "paquets_annees": "integer ou null"
                    },
                    "alcool": {"quantite_par_frequence": "string ou null"},
                    "activite_physique": [{"type": "string ou null", "frequence": "string ou null"}]
                },
                "vaccins": [{
                    "nom_commercial": "string ou null",
                    "date_dose_1": "string ISO 8601 ou null",
                    "rappel_prevu": "string ISO 8601 ou null"
                }]
            },
            "documents": {
                "consultations": [{
                    "date": "string ISO 8601 ou null",
                    "medecin": {
                        "nom": "string ou null",
                        "prenom": "string ou null",
                        "specialite": "string ou null"
                    },
                    "motif_de_consultation": "string ou null",
                    "interrogatoire": "string ou null",
                    "examen_clinique": "string ou null",
                    "conclusion": "string ou null"
                }]
            }
        }
    }

    if existing_dpi:
        dpi_context = f"""
Le DPI existant du patient est fourni ci-dessous. Enrichis-le avec les informations
extraites de la transcription. Conserve toutes les données existantes et ajoute ou
mets à jour uniquement ce qui est mentionné dans la transcription.
Ajoute la nouvelle consultation à la liste dpi.documents.consultations.

DPI EXISTANT :
{json.dumps(existing_dpi, ensure_ascii=False, indent=2)}
"""
    else:
        dpi_context = """
Aucun DPI existant. Construis le DPI complet à partir des informations extraites
de la transcription. Les champs non mentionnés doivent être null.
"""

    return f"""Tu es un assistant médical expert en extraction d'informations cliniques.
{dpi_context}

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

INSTRUCTIONS :
- Extrais toutes les informations médicales présentes dans la transcription.
- La nouvelle entrée dans dpi.documents.consultations doit contenir :
    motif_de_consultation, interrogatoire (symptômes, histoire de la maladie),
    examen_clinique (constantes, examen physique), conclusion (diagnostic,
    proposition thérapeutique, prochaine consultation).
- Les champs non mentionnés dans la transcription doivent être null, jamais inventés.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RESPECTER :
{json.dumps(dpi_schema, ensure_ascii=False, indent=2)}
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