"""
geo_tagging.py
--------------
Extracts a geographic-source tag ("BO", "PBO", "EmiliaR", "NorthEast", "Italy")
for every marginal and CPT table in attr_meta_ISTAT.py, by parsing the inline
comments that are already there (e.g. "#BO", "#EmiliaR", "#NorthEast", or a
free-text comment like "EmiliaR Labour Force and EmiliaR PartTimeFullTime").

Anything with no recognisable tag defaults to "Italy" (national ISTAT figure),
which is the most conservative assumption: untagged tables get the LOWEST
reliability weight unless you correct them by hand.

This module does not change any modelling code. It only produces:
    TABLE_SOURCE   : dict  {cpt_table_name  -> geo tag}
    MARGINAL_SOURCE: dict  {attribute_name  -> geo tag}
    RELIABILITY    : dict  {geo tag -> weight in (0, 1]}

You should treat RELIABILITY as a hyperparameter, not as a physical constant.
The values below are a reasonable starting point (roughly: how many "levels
of aggregation" removed from the target population, Comune di Bologna).
Re-tune them once you have a stratified-MRE readout (see mre_diagnostics.py).

IMPORTANT: the auto-parser is a best-effort regex scan of your comments.
Always inspect TABLE_SOURCE / MARGINAL_SOURCE before trusting them -- print
the ones tagged "Italy" and check by hand that they really are untagged
national figures and not just missing a comment.
"""

import re
from collections import OrderedDict

# ------------------------------------------------------------------ #
#  Reliability weights per geographic level                            #
#  (edit these -- they are the W_jj of the soft-constraint objective)  #
# ------------------------------------------------------------------ #
RELIABILITY = OrderedDict([
    ("BO",        1.00),   # Comune di Bologna -- the actual target population
    ("PBO",       0.85),   # Provincia / metropolitan area of Bologna
    ("EmiliaR",   0.50),   # Regione Emilia-Romagna
    ("NorthEast", 0.30),   # Macro-area Nord-Est
    ("Italy",     0.15),   # National ISTAT figure / untagged default
])

# Aliases -> canonical tag. Order matters: longer/more specific first.
_ALIASES = [
    (r"\bPBO\b",                              "PBO"),
    (r"\bBO\s*20\d\d\b",                      "BO"),
    (r"\bBO\b",                               "BO"),
    (r"\bBologna\b",                          "BO"),
    (r"\bEmilia[\s\-]?R(omagna)?\b",          "EmiliaR"),
    (r"\bNorth[\s\-]?East\b",                 "NorthEast"),
    (r"\bNordEst\b",                          "NorthEast"),
    (r"\b#?IT\b",                             "Italy"),   # shorthand #IT used in attr_meta_ISTAT
]

_TAG_RE = re.compile("|".join(p for p, _ in _ALIASES))


def _extract_tag(comment_text: str) -> str | None:
    """Return the first recognised geo tag found in a comment string, or None."""
    if not comment_text:
        return None
    for pattern, tag in _ALIASES:
        if re.search(pattern, comment_text, flags=re.IGNORECASE):
            return tag
    return None


def _strip_code_get_comment(line: str) -> str:
    """Return the '#...' trailing comment portion of a line (outside of quotes)."""
    in_str = False
    quote_char = ""
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            if not in_str:
                in_str, quote_char = True, ch
            elif ch == quote_char:
                in_str = False
        elif ch == "#" and not in_str:
            return line[i:]
    return ""


def parse_source_tags(filepath: str, default_tag: str = "Italy"):
    """
    Scan attr_meta_ISTAT.py and return (table_source, marginal_source):

      table_source   : {cpt_table_variable_name -> geo tag}
      marginal_source: {attribute_name          -> geo tag}

    Heuristics (best-effort, always spot-check the output):
      * CPT tables: look at the trailing comment on the `name = {` line,
        and the full text of the '#'-comment line immediately above it
        (this is where "# B9: P(Sex | Employment) EmiliaR ..." style
        annotations live).
      * marginals dict: look at the trailing comment on every line that
        contains `"attr_name":` inside the `marginals = { ... }` block.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    table_source = {}
    marginal_source = {}

    table_def_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{")
    marginal_key_re = re.compile(r'^\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:')

    in_marginals = False
    marginals_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- track whether we're inside the top-level `marginals = {...}` dict ---
        if re.match(r"^marginals\s*=\s*\{", stripped):
            in_marginals = True
            marginals_depth = line.count("{") - line.count("}")
            continue
        if in_marginals:
            marginals_depth += line.count("{") - line.count("}")
            m = marginal_key_re.match(line)
            if m:
                attr = m.group(1)
                tag = _extract_tag(_strip_code_get_comment(line))
                if tag:
                    marginal_source[attr] = tag
            if marginals_depth <= 0:
                in_marginals = False
            continue

        # --- CPT table definitions: `name = {` at column 0 ---
        m = table_def_re.match(line)
        if m and not stripped.startswith(("ATTR_META", "marginals")):
            name = m.group(1)
            same_line_comment = _strip_code_get_comment(line)
            # Scan back up to 4 lines to find the nearest preceding comment
            # (some tables have blank lines between the # BN: ... comment and the def)
            prev_line_comment = ""
            for look_back in range(1, 5):
                if i >= look_back:
                    candidate = lines[i - look_back].strip()
                    if candidate.startswith("#"):
                        prev_line_comment = lines[i - look_back]
                        break
                    elif candidate == "":
                        continue   # blank line — keep scanning back
                    else:
                        break      # hit code — stop
            tag = _extract_tag(same_line_comment) or _extract_tag(prev_line_comment)
            if tag:
                table_source[name] = tag

    # fill defaults for anything referenced later but not found
    return table_source, marginal_source


def build_reliability_lookup(table_source: dict, marginal_source: dict,
                              default_tag: str = "Italy"):
    """
    Convenience wrapper: returns two dicts (table_name/attr_name -> weight)
    ready to feed into the constraint-weight builder in preprocess_istat.py.
    """
    table_weight = {name: RELIABILITY.get(tag, RELIABILITY[default_tag])
                     for name, tag in table_source.items()}
    marginal_weight = {attr: RELIABILITY.get(tag, RELIABILITY[default_tag])
                        for attr, tag in marginal_source.items()}
    return table_weight, marginal_weight


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "attr_meta_ISTAT.py"
    tbl, marg = parse_source_tags(path)

    print(f"Tagged {len(tbl)} CPT tables, {len(marg)} marginals with an explicit geo source.\n")
    print("── CPT tables ──────────────────────────────────────")
    for name, tag in sorted(tbl.items()):
        print(f"  {name:<38} {tag}")
    print("\n── Marginals ────────────────────────────────────────")
    for name, tag in sorted(marg.items()):
        print(f"  {name:<38} {tag}")

    print("\n(Everything not listed above defaults to 'Italy' -- national ISTAT "
          "figure with no explicit local tag. Verify this assumption by hand.)")