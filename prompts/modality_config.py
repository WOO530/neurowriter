"""Modality-specific configuration for EEG, PSG, and mixed contexts.

Centralises all modality-conditional data (keywords, section names,
few-shot examples, query templates) so that prompt modules can
select the right context without scattering if/else logic.
"""

from typing import Dict, List


# ------------------------------------------------------------------
# Modality detection keywords
# ------------------------------------------------------------------

MODALITY_KEYWORDS: Dict[str, List[str]] = {
    "eeg": [
        "eeg", "electroencephalogra", "event-related potential", "erp",
        "spectral power", "oscillat", "alpha band", "beta band",
        "gamma band", "theta band", "delta band",
        "p300", "n100", "n200", "mismatch negativity", "mmn",
        "qeeg", "brain-computer interface", "bci",
    ],
    "psg": [
        "psg", "polysomnograph", "sleep stage", "sleep study",
        "sleep architecture", "rem sleep", "nrem", "slow wave sleep",
        "apnea", "hypopnea", "ahi", "oxygen desaturation",
        "narcolepsy", "insomnia", "rem behavior disorder", "rbd", "irbd",
        "circadian", "sleep spindle", "k-complex", "arousal index",
        "periodic limb movement", "plm", "restless legs",
        "obstructive sleep apnea", "osa", "central sleep apnea",
        "sleep efficiency", "sleep latency", "wake after sleep onset",
        "sleep fragmentation", "sleep disorder",
        "cpap", "sleep medicine",
        "mslt", "multiple sleep latency", "maintenance of wakefulness",
        "mwt", "idiopathic hypersomnia",
    ],
}


def detect_modality(research_topic: str, topic_analysis: Dict = None) -> str:
    """Detect whether the research topic is EEG-focused, PSG-focused, or mixed.

    Args:
        research_topic: Raw research topic string
        topic_analysis: Parsed topic analysis (optional, for extra signal)

    Returns:
        "eeg", "psg", or "mixed"
    """
    text = research_topic.lower()
    if topic_analysis:
        text += " " + (topic_analysis.get("data_type", "") or "").lower()
        text += " " + (topic_analysis.get("disease", "") or "").lower()
        text += " " + (topic_analysis.get("key_intervention_or_focus", "") or "").lower()

    eeg_hits = sum(1 for kw in MODALITY_KEYWORDS["eeg"] if kw in text)
    psg_hits = sum(1 for kw in MODALITY_KEYWORDS["psg"] if kw in text)

    if eeg_hits > 0 and psg_hits > 0:
        return "mixed"
    elif psg_hits > 0:
        return "psg"
    else:
        return "eeg"  # default to EEG (preserves existing behaviour)


# ------------------------------------------------------------------
# Section names by modality
# ------------------------------------------------------------------

SECTION_NAMES: Dict[str, Dict[str, str]] = {
    "eeg": {
        "biomarker_section": "Neurophysiological Biomarkers",
        "domain_label": "neuroscience, neurology, and psychiatry",
        "condition_label": "psychiatric conditions",
        "journal_examples": "JAMA Psychiatry, JAMA Neurology, Lancet Neurology, Lancet Psychiatry, Biological Psychiatry",
    },
    "psg": {
        "biomarker_section": "Polysomnographic Biomarkers",
        "domain_label": "sleep medicine, neurology, and neuroscience",
        "condition_label": "sleep disorders",
        "journal_examples": "Sleep, JAMA Neurology, Annals of Neurology, Sleep Medicine Reviews, Lancet Neurology",
    },
    "mixed": {
        "biomarker_section": "Neurophysiological and Sleep Biomarkers",
        "domain_label": "neuroscience, neurology, psychiatry, and sleep medicine",
        "condition_label": "neuropsychiatric and sleep disorders",
        "journal_examples": "JAMA Psychiatry, JAMA Neurology, Lancet Neurology, Sleep, Biological Psychiatry",
    },
}


# ------------------------------------------------------------------
# Few-shot academic examples by modality
# ------------------------------------------------------------------

ACADEMIC_EXAMPLES: Dict[str, str] = {
    "eeg": """EXAMPLE PARAGRAPH STYLES (for tone reference, do NOT cite):

Example 1 - Disease Background:
"Major depressive disorder (MDD) is a prevalent and debilitating psychiatric condition, affecting approximately 280 million individuals globally [1] and ranking among the leading causes of disability worldwide [2]. Despite the availability of multiple pharmacological interventions, treatment outcomes remain suboptimal [3], with approximately one-third of patients failing to achieve adequate remission following first-line antidepressant therapy [4-5]."

Example 2 - Technology/Methodology:
"Electroencephalography (EEG), a non-invasive neurophysiological modality with high temporal resolution, has emerged as a promising tool for identifying neural biomarkers associated with treatment response in psychiatric disorders [6-8]. Recent advances in deep learning architectures, particularly convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have demonstrated remarkable capacity for extracting complex spatiotemporal features from raw EEG signals that may elude conventional analytical approaches [9,10]."

Example 3 - Unmet Need/Rationale:
"Although substantial progress has been made in characterizing EEG abnormalities in MDD [11], the translation of these findings to clinical practice has been limited [12]. Current diagnostic procedures rely exclusively on clinical assessment, lacking objective biological biomarkers [13]. The identification of reliable EEG-based predictors of antidepressant response could substantially improve treatment selection and outcomes [14], representing a significant unmet clinical need." """,

    "psg": """EXAMPLE PARAGRAPH STYLES (for tone reference, do NOT cite):

Example 1 - Disease Background:
"Obstructive sleep apnea (OSA) is a highly prevalent sleep-related breathing disorder characterized by repetitive upper airway collapse during sleep, affecting an estimated 936 million adults globally [1] and representing a substantial burden on healthcare systems worldwide [2]. Despite its high prevalence, OSA remains underdiagnosed in clinical practice [3], with up to 80% of moderate-to-severe cases undetected in the general population [4,5]."

Example 2 - Technology/Methodology:
"Polysomnography (PSG), the gold standard for diagnosing sleep disorders, generates multimodal physiological recordings including electroencephalography, electromyography, electrooculography, and respiratory signals that collectively provide comprehensive characterization of sleep architecture and pathology [6-8]. The application of deep learning techniques to PSG data has shown considerable promise in automating sleep staging [9], with convolutional and recurrent neural network architectures achieving expert-level accuracy in classifying sleep stages from raw signals [10]."

Example 3 - Unmet Need/Rationale:
"Although automated sleep staging using machine learning has advanced substantially [11], the prediction of clinical outcomes such as treatment response or disease severity from PSG features remains limited [12]. Current clinical decision-making relies predominantly on summary indices such as the apnea-hypopnea index (AHI), which may not fully capture the heterogeneity of sleep pathology [13]. The development of PSG-based predictive models could enable more personalized therapeutic strategies [14], addressing a critical gap in sleep medicine." """,

    "mixed": """EXAMPLE PARAGRAPH STYLES (for tone reference, do NOT cite):

Example 1 - Disease Background:
"Major depressive disorder (MDD) is a prevalent and debilitating psychiatric condition, affecting approximately 280 million individuals globally [1] and ranking among the leading causes of disability worldwide [2]. Sleep disturbances, present in up to 90% of MDD patients [3], represent both a core symptom and a potential marker of treatment response [4,5]."

Example 2a - Technology/Methodology (EEG):
"Electroencephalography (EEG), a non-invasive neurophysiological modality with high temporal resolution, has emerged as a promising tool for identifying neural biomarkers in psychiatric disorders [6-8]."

Example 2b - Technology/Methodology (PSG):
"Polysomnography (PSG) provides complementary characterization of sleep architecture abnormalities, including alterations in REM latency and slow wave sleep that have been consistently observed in mood disorders [9,10]."

Example 3 - Unmet Need/Rationale:
"The integration of waking EEG biomarkers with sleep architecture features from PSG could provide a more comprehensive neurophysiological profile for predicting treatment outcomes [11,12], yet such multimodal approaches remain largely unexplored [13]." """,
}


# ------------------------------------------------------------------
# Topic parsing query examples by modality
# ------------------------------------------------------------------

QUERY_EXAMPLES: Dict[str, str] = {
    "eeg": """    "search_queries": [
        "(major depressive disorder OR MDD) treatment outcomes",
        "(major depressive disorder OR MDD) prevalence global burden disability",
        "(treatment resistant depression OR TRD) antidepressant non-response",
        "(electroencephalography OR EEG) biomarker (major depressive disorder OR MDD)",
        "(EEG OR electroencephalography) (deep learning OR neural network) classification",
        "alpha asymmetry theta power frontal EEG depression",
        "(EEG OR electroencephalography) (machine learning OR deep learning) antidepressant treatment response prediction",
        "neurophysiological biomarker treatment selection psychiatry",
        "(convolutional neural network OR recurrent neural network OR transformer) EEG signal classification",
        "EEG clinical decision support psychiatric diagnosis",
        "frontal theta cordance antidepressant mechanism EEG",
        "(event-related potential OR P300 OR mismatch negativity) depression",
        "(EEG OR electroencephalography) AND (deep learning OR machine learning) AND depression AND (systematic review OR meta-analysis)",
        "resting state EEG connectivity functional brain network depression",
        "personalized treatment prediction biomarker (major depressive disorder OR MDD)"
    ]""",

    "psg": """    "search_queries": [
        "(obstructive sleep apnea OR OSA) treatment outcomes",
        "(obstructive sleep apnea OR OSA) prevalence cardiovascular morbidity",
        "(positional obstructive sleep apnea OR OSA phenotype)",
        "(polysomnography OR PSG) (obstructive sleep apnea OR OSA) diagnosis",
        "(polysomnography OR PSG) (deep learning OR neural network) automated sleep staging",
        "sleep architecture arousal index oxygen desaturation",
        "(sleep apnea OR OSA) severity prediction (machine learning OR deep learning)",
        "(polysomnography OR PSG) features (CPAP OR treatment) response prediction",
        "sleep spindle slow wave activity biomarker cognitive decline",
        "home sleep apnea testing versus (polysomnography OR PSG) accuracy",
        "intermittent hypoxia oxidative stress (sleep apnea OR OSA) pathophysiology",
        "(sleep apnea OR OSA) (depression OR anxiety) (polysomnography OR PSG)",
        "(sleep apnea OR OSA) AND (deep learning OR machine learning) AND (systematic review OR meta-analysis)",
        "(convolutional neural network OR recurrent neural network) PSG signal classification sleep disorder",
        "AHI limitations (sleep apnea OR OSA) heterogeneity phenotyping (polysomnography OR PSG)"
    ]""",
}
