"""Genre profiles for different document types.

Each profile defines the role, narrative structure guide, and content
proportion constraints that shape how the writing strategy is generated.
Add new entries to GENRE_PROFILES to support additional document types
(e.g. grant backgrounds, pharma proposals).
"""

from typing import Dict

GENRE_PROFILES: Dict[str, dict] = {
    "research_introduction": {
        "label": "Research Paper Introduction",
        "role": (
            "You are an expert medical research writer planning a research paper "
            "introduction in {domain_label}, for a journal such as {journal_examples}.\n\n"
            "You are writing a CLINICAL INTRODUCTION — the kind published in top "
            "medical journals.\n"
            "This is NOT a methods paper, technical report, or project proposal."
        ),
        "structure_guide": (
            "NARRATIVE FLOW GUIDE:\n"
            "Your introduction must tell a clinical story that justifies the study.\n"
            "Typical effective flows in medical research introductions:\n\n"
            "  Clinical context & significance (disease burden, epidemiology, why this\n"
            "  condition matters — its impact on patients, healthcare systems, or society)\n"
            "  → Prior research landscape (classical biomarker findings, statistical analyses,\n"
            "    and modern computational approaches — woven into a coherent narrative,\n"
            "    with key results and their limitations)\n"
            "  → Current approach rationale (why the proposed modality and methodology\n"
            "    can address remaining gaps; advantages over prior approaches)\n"
            "  → Unmet need & study aims (what gap remains, what this study will do)\n\n"
            "You may merge or split these elements across paragraphs as the topic demands.\n"
            "The exact number of paragraphs (3-5) and their boundaries are flexible."
        ),
        "proportion_guide": (
            "CONTENT PROPORTION CONSTRAINTS (based on published medical introductions):\n"
            "- Clinical context (disease burden + treatment/diagnostic limitations): "
            "≥35% of total content\n"
            "- Methodology/technical explanation should NOT dominate: ≤30% of total content\n"
            "- Study aims paragraph: concise, ~10-15% of total content\n"
            "- Every paragraph must serve the clinical narrative — avoid standalone "
            "method tutorials"
        ),
    },
    # Future genres:
    # "grant_background": { ... },
    # "pharma_proposal": { ... },
}
