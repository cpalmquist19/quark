"""Domain filters for data mixtures (plan §18, docs/domain_data_survey.md).

A document belongs to a domain when the domain is one of its subjects, not a passing mention: at least
`min_hits` keyword matches, `min_distinct` different terms, and `min_per_1k_words` matches per 1,000
words. The defaults are the survey's "strict" threshold. Curation (sites or phrasings to leave out)
lives in the mix recipe, where it can be read and changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

DOMAINS: dict[str, list[str]] = {
    "neurodivergence": [
        r"autism", r"autistic", r"asperger'?s?", r"adhd", r"attention[- ]deficit", r"dyslexi[ac]", r"dyscalculia",
        r"dyspraxia", r"dysgraphia", r"neurodivergen(?:t|ce)", r"neurodiversity", r"neurotypical",
        r"sensory processing", r"executive function(?:ing|s)?", r"stimming", r"on the (?:autism )?spectrum",
        r"learning disabilit(?:y|ies)", r"developmental (?:delay|disabilit(?:y|ies))", r"tourette'?s?",
        r"special needs", r"occupational therap(?:y|ist)", r"speech therap(?:y|ist)", r"iep",
        r"individuali[sz]ed education(?:al)? (?:program|plan)", r"sensory overload", r"social skills",
    ],
    "education": [
        r"lesson plans?", r"classrooms?", r"curricul(?:um|a)", r"pedagog(?:y|ical)", r"teachers?",
        r"instructional", r"differentiated instruction", r"formative assessment", r"learning objectives?",
        r"scaffold(?:ing)?", r"homeschool(?:ing)?", r"student engagement", r"literacy", r"special education",
        r"educators?", r"teaching strateg(?:y|ies)", r"learning outcomes?",
    ],
    "early_childhood": [
        r"preschool(?:ers?)?", r"pre-?k", r"kindergarten", r"toddlers?", r"early childhood", r"early years",
        r"nursery school", r"phonics", r"alphabet", r"letter sounds?", r"sight words?", r"fine motor", r"gross motor",
        r"play-based", r"social[- ]emotional", r"circle time", r"young children", r"daycare", r"day care",
        r"early literacy", r"(?:3|4|5|three|four|five)[- ]year[- ]olds?", r"potty", r"nap ?time", r"storytime",
        r"counting to (?:10|ten|20|twenty)", r"colors and shapes", r"shapes and colou?rs",
    ],
}


def _words(terms: list[str]) -> re.Pattern:
    return re.compile("|".join(rf"\b(?:{t})\b" for t in terms), re.I)


@dataclass
class DocFilter:
    """Decides whether a streamed row belongs to a source. Every condition given must hold."""

    domain: str | None = None
    min_hits: int = 3
    min_distinct: int = 2
    min_per_1k_words: float = 2.0
    columns: dict[str, list[str]] | None = None  # row[column] must be one of these values
    exclude_sites: tuple[str, ...] = ()  # hostnames (and their subdomains) to leave out
    exclude_patterns: tuple[str, ...] = ()  # regexes; any match leaves the document out

    def __post_init__(self):
        if self.domain is not None and self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r} (known: {', '.join(DOMAINS)})")
        self._domain = _words(DOMAINS[self.domain]) if self.domain else None
        self._exclude = re.compile("|".join(f"(?:{p})" for p in self.exclude_patterns), re.I) if self.exclude_patterns else None
        self._sites = tuple(s.lower().removeprefix("www.") for s in self.exclude_sites)

    def accepts(self, row: dict, text_field: str = "text") -> bool:
        if self.columns:
            for column, allowed in self.columns.items():
                if row.get(column) not in allowed:
                    return False
        if self._sites and (url := row.get("url")):
            host = urlparse(url).netloc.lower().removeprefix("www.")
            if any(host == s or host.endswith("." + s) for s in self._sites):
                return False
        text = row[text_field]
        if self._domain is not None and not self.is_about(text):
            return False
        return not (self._exclude and self._exclude.search(text))

    def is_about(self, text: str) -> bool:
        found = [m.group(0).lower() for m in self._domain.finditer(text)]
        words = max(1, len(text.split()))
        return (
            len(found) >= self.min_hits
            and len(set(found)) >= self.min_distinct
            and len(found) * 1000 / words >= self.min_per_1k_words
        )
