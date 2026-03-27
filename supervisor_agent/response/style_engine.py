"""Dynamic CSS style engine for report generation.

Owns all CSS logic: theme presets, base stylesheet, style extraction from
user queries, and LLM-powered CSS generation for freeform styling requests.

Architecture: base CSS (layout/spacing) + theme preset (brand colors/fonts)
+ dynamic CSS overrides (LLM-generated from user instructions, !important).
"""

import logging
import re
from typing import Dict

from ..llm_provider import llm_complete

logger = logging.getLogger(__name__)

# ── Layer 1: Theme presets — safe defaults for colors and fonts ──

THEMES: Dict[str, Dict[str, str]] = {
    "default": {
        "--color-primary": "#028090",
        "--color-accent": "#02c39a",
        "--font-family": "'Segoe UI', system-ui, -apple-system, sans-serif",
        "--header-bg": "#028090",
        "--header-text": "#ffffff",
    },
    "clinical": {
        "--color-primary": "#1a365d",
        "--color-accent": "#2b6cb0",
        "--font-family": "Georgia, 'Times New Roman', serif",
        "--header-bg": "#1a365d",
        "--header-text": "#ffffff",
    },
    "minimal": {
        "--color-primary": "#333333",
        "--color-accent": "#666666",
        "--font-family": "'Helvetica Neue', Arial, sans-serif",
        "--header-bg": "#f0f0f0",
        "--header-text": "#333333",
    },
}

# ── Layer 2: Base CSS template — layout, spacing, typography ──

BASE_CSS = """
:root { %(vars)s }
@page { size: %(page_size)s; margin: 2cm; @bottom-center { content: counter(page); } }
body { font-family: var(--font-family); font-size: 11pt; line-height: 1.5; color: #222;
       orphans: 2; widows: 2; }
h1 { color: var(--color-primary); border-bottom: 2px solid var(--color-accent);
     padding-bottom: 6px; margin-top: 0; }
h2 { color: var(--color-primary); margin-top: 1.6em; margin-bottom: 0.4em;
     page-break-after: avoid; }
h3 { color: #444; margin-top: 1.2em; page-break-after: avoid; }
table { width: 100%%; border-collapse: collapse; margin: 1em 0; font-size: 9pt;
        page-break-inside: avoid; }
th { background: var(--header-bg); color: var(--header-text); padding: 6px 8px;
     text-align: left; border: 1px solid #ccc; }
td { padding: 5px 8px; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f8f9fa; }
pre { background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 9pt;
      page-break-inside: avoid; }
code { font-size: 9pt; }
.report-header { text-align: center; margin-bottom: 1.5em; }
.report-header .disease { color: var(--color-accent); font-style: italic; }
.report-header .dateline { color: #888; font-size: 10pt; }
.report-body { max-width: 100%%; }
.report-footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid #ccc;
                 font-size: 9pt; color: #666; text-align: center; }
.section-divider { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
blockquote { border-left: 3px solid var(--color-accent); margin: 1em 0;
             padding: 0.5em 1em; background: #f9fafb; }
@media print {
  body { font-size: 10pt; }
  h2, h3 { page-break-after: avoid; }
  table, figure, blockquote { page-break-inside: avoid; }
}
"""""

# ── Layer 3: Dynamic style extraction + LLM CSS generation ──

_FUNCTIONAL_WORDS = re.compile(
    r'\b(can you|please|generate|create|make|produce|give|send|build|'
    r'a|an|the|with|and|or|is|are|it|its|this|that|those|'
    r'pdf|docx?|word|report|document|file|csv|'
    r'top|genes?|pathways?|drugs?|results?|data|'
    r'summary|overview|analysis|enrichment|prioritized|uploaded|'
    r'for|my|me|of|in|on|using|about|based|from)\b',
    re.IGNORECASE,
)

# Allowlist of terms that indicate genuine styling intent.
# Used as a gate: carry-forward only triggers if residual text contains these.
_STYLE_KEYWORDS = re.compile(
    r'\b('
    # colours
    r'red|blue|green|black|white|gray|grey|navy|dark|light|teal|orange|'
    r'purple|yellow|pink|brown|maroon|cyan|magenta|indigo|violet|gold|'
    # colour modifiers
    r'colou?r|shade|tone|palette|accent|background|bg|'
    # typography
    r'font|serif|sans[- ]?serif|monospace|bold|italic|underline|'
    r'heading|title|text|size|larger|smaller|bigger|tiny|huge|'
    # layout
    r'compact|wide|narrow|two[- ]?column|single[- ]?column|landscape|portrait|'
    r'margin|padding|spacing|indent|center|left|right|align|'
    # theming
    r'theme|style|mode|dark[- ]?mode|light[- ]?mode|clinical|minimal|modern|classic|'
    r'corporate|elegant|professional|clean|nature[- ]?style|science[- ]?style|'
    # visual elements
    r'border|line|divider|separator|table[- ]?style|grid|striped|'
    r'icon|logo|watermark|header|footer'
    r')\b',
    re.IGNORECASE,
)

CSS_GENERATION_PROMPT = """\
You are a CSS expert generating style overrides for biomedical analysis \
reports rendered as HTML documents. Given the user's styling instructions, \
produce a CSS snippet that transforms the report's visual appearance.

═══ DOCUMENT STRUCTURE YOU'RE STYLING ═══

The HTML document has this general structure:
- <body> containing the full report
- <div class="report-header"> with <h1> title, <span class="disease"> \
  for disease context, <span class="dateline"> for generation date
- <main class="report-body"> wrapping all content sections
- <h2> for major sections, <h3> for subsections
- <table> for data tables, with <thead>/<tbody>/<th>/<td>
- <p> for body text
- <strong>, <em> for inline emphasis
- <code> for gene/pathway identifiers
- <hr class="section-divider"> for section dividers
- <div class="report-footer"> at the bottom
- <blockquote> for highlighted quotes or key statistics

═══ GENERATION RULES ═══

1. Output ONLY valid CSS. No explanations, no markdown, no code fences, \
   no comments unless they aid readability of complex selectors.
2. Use !important on every property to ensure overrides take effect \
   over the base theme.
3. Only override what the user requested. If they say "red headings", \
   only change h1, h2, h3 color — do not touch anything else.
4. Use hex color codes. If the user names a color, map it intelligently:
   - Named colors → appropriate professional hex values
   - "Dark mode" → #1a1a2e body background, #e2e8f0 text, \
     adjusted table/heading/border colors for contrast
   - "Light mode" → #ffffff body, #111111 text
   - "Corporate" → navy headers, gray accents, clean sans-serif
   - "Nature style" → muted earth tones, serif body font, \
     minimal table borders
   - "Clinical" → clean white, blue accents, high-contrast tables
   For any color name not listed, choose a professional, accessible hex.
5. For font requests: use web-safe font stacks. \
   "Modern" → 'Inter', 'Segoe UI', system-ui, sans-serif \
   "Classic" → 'Georgia', 'Cambria', 'Times New Roman', serif \
   "Monospace" → 'JetBrains Mono', 'Fira Code', 'Consolas', monospace
6. For layout requests ("two columns", "wider tables", "compact"): \
   use appropriate CSS (grid, flexbox, margin/padding adjustments).
7. Respect print media: if the document will be rendered as PDF, \
   avoid viewport-dependent units (vh, vw). Use pt, mm, cm, or rem.
8. Ensure sufficient color contrast (WCAG AA minimum) — if the user \
   asks for light text on light background, choose a darker text shade \
   automatically and note this in a CSS comment.
9. For complex style requests ("make it look like a Nature paper", \
   "biotech startup style", "clinical trial report format"):
   - Interpret the intent and generate comprehensive CSS covering \
     typography, colors, spacing, table styling, and header treatment.
   - These are multi-property requests — generate 20-40 lines of CSS.
10. For partial/ambiguous requests ("make it prettier", "improve the style"):
    - Apply a tasteful professional upgrade: slightly tighter line-height, \
      better heading spacing, subtle table alternating row colors, \
      refined border treatment. Keep changes conservative.

═══ INCREMENTAL STYLING ═══

The user may provide multiple styling instructions across conversation turns. \
Each CSS generation is additive — you're producing overrides that layer on top \
of previous styles. Generate ONLY the new/changed properties for this request, \
not a complete re-theme (unless the user says "start over" or "reset styling")."""


def has_style_intent(query: str) -> bool:
    """Return True if the query contains genuine styling keywords."""
    return bool(_STYLE_KEYWORDS.search(query))


def extract_style_instructions(query: str, fmt: str, disease: str = "") -> str:
    """Extract styling intent from user query via zero-latency regex heuristic.

    Returns non-empty only when the query contains genuine style keywords
    (colours, fonts, layout, theming). Scientific domain terms are NOT
    style instructions — the allowlist gate prevents false positives.
    """
    if fmt not in ("pdf", "docx"):
        return ""
    # Gate: query must contain at least one style keyword to proceed
    if not _STYLE_KEYWORDS.search(query):
        return ""
    cleaned = _FUNCTIONAL_WORDS.sub("", query)
    # Strip disease name (avoids "sarcoidosis" leaking into style instructions)
    if disease:
        cleaned = re.sub(re.escape(disease), "", cleaned, flags=re.IGNORECASE)
    # Strip standalone numbers ("top 20" → "20" leak)
    cleaned = re.sub(r"\b\d+\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) > 3 else ""


def _balance_braces(css: str) -> str:
    """Truncate CSS at the last balanced closing brace if braces are unmatched."""
    depth, last_balanced = 0, -1
    for i, ch in enumerate(css):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_balanced = i
    if depth != 0 and last_balanced >= 0:
        return css[: last_balanced + 1]
    return css


def sanitize_css(css: str) -> str:
    """Strip dangerous constructs and fix truncated rules in LLM-generated CSS."""
    css = re.sub(r"<script[^>]*>.*?</script>", "", css, flags=re.DOTALL | re.IGNORECASE)
    css = re.sub(r"@import\b[^;]*;?", "", css, flags=re.IGNORECASE)
    css = re.sub(r"url\s*\(", "/* blocked-url(", css, flags=re.IGNORECASE)
    css = re.sub(r"expression\s*\(", "/* blocked-expr(", css, flags=re.IGNORECASE)
    css = re.sub(r"javascript\s*:", "/* blocked */", css, flags=re.IGNORECASE)
    # Strip markdown code fences the LLM might wrap output in
    css = re.sub(r"^```\w*\s*", "", css.strip())
    css = re.sub(r"\s*```$", "", css.strip())
    css = _balance_braces(css.strip())
    return css.strip()


async def generate_style_css(style_instructions: str, base_theme: str = "default") -> str:
    """Generate a CSS override snippet from freeform user styling instructions."""
    if not style_instructions:
        return ""
    theme_desc = f"Current base theme: {base_theme} ({THEMES.get(base_theme, THEMES['default'])})"
    try:
        result = await llm_complete(
            messages=[{"role": "user", "content": f"{theme_desc}\n\nUser request: {style_instructions}"}],
            system=CSS_GENERATION_PROMPT,
            max_tokens=600,
            temperature=0.1,
        )
        css = sanitize_css(result.text)
        logger.info(f"style_engine: generated {len(css)} chars CSS for '{style_instructions}'")
        return css
    except Exception as exc:
        logger.warning(f"style_engine: CSS generation failed ({exc}), using base theme")
        return ""


def build_css(theme: str = "default", page_size: str = "A4", custom_css: str = "") -> str:
    """Assemble the complete CSS: base template + theme variables + dynamic overrides."""
    theme_vars = THEMES.get(theme, THEMES["default"])
    css_vars = " ".join(f"{k}: {v};" for k, v in theme_vars.items())
    css = BASE_CSS % {"vars": css_vars, "page_size": page_size}
    if custom_css:
        css += "\n/* ── Dynamic overrides ── */\n" + custom_css
    return css
