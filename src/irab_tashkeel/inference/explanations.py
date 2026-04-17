"""Bilingual (Arabic + English) labels for i'rab roles and error types."""

ROLE_LABELS_AR = {
    "other":        "—",
    "fiil":         "فعل",
    "harf_jarr":    "حرف جر",
    "harf_atf":     "حرف عطف",
    "harf_nafy":    "حرف نفي",
    "mabni_noun":   "اسم مبني",
    "N_marfu":      "اسم مرفوع",
    "N_mansub":     "اسم منصوب",
    "ism_majrur":   "اسم مجرور",
    "mudaf_ilayh":  "مضاف إليه",
    "<pad>":        "",
}

ROLE_LABELS_EN = {
    "other":        "other",
    "fiil":         "verb",
    "harf_jarr":    "preposition",
    "harf_atf":     "coordinator",
    "harf_nafy":    "negation",
    "mabni_noun":   "indeclinable noun",
    "N_marfu":      "nominative noun",
    "N_mansub":     "accusative noun",
    "ism_majrur":   "preposition object (gen)",
    "mudaf_ilayh":  "iḍāfa complement (gen)",
    "<pad>":        "",
}

ERROR_DESCRIPTIONS = {
    "hamza": "Hamza qaṭʿ missing — should have أ / إ / آ",
    "taa":   "Tāʾ marbūṭa should be ة, not ه",
    "case":  "Case ending inconsistent with grammatical role",
}


def role_to_ar(role: str) -> str:
    return ROLE_LABELS_AR.get(role, role)


def role_to_en(role: str) -> str:
    return ROLE_LABELS_EN.get(role, role)


def describe_error(err_type: str) -> str:
    return ERROR_DESCRIPTIONS.get(err_type, err_type)
