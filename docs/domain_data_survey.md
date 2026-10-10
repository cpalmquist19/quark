# Domain data survey (2026-10-10)

Input for the data mixture (plan §18). The question: how much text exists for each target domain, what is it
like, and what does that mean for the mix?

## Method

- **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, subset `sample-10BT`): the first 120,000 streamed documents
  (125.8M tokens, GPT-2 token counts from the dataset), scanned with per-domain keyword patterns
  (case-insensitive, whole words).
  - **Loose:** the document mentions the domain at least once.
  - **Strict:** at least 3 matches, at least 2 distinct terms, and at least 2 matches per 1,000 words. In other
    words, the domain is a subject of the document, not a passing mention.
- **Cosmopedia** (`HuggingFaceTB/cosmopedia`): the Hugging Face dataset-viewer statistics for the `audience` and
  `format` columns. They're computed on a partial sample, so the shares below are scaled by each subset's full row
  count. Plus a reading of streamed samples.
- Keyword lists (to become the mixture's domain filters):
  - **Neurodivergence:** autism/autistic, Asperger's, ADHD, attention deficit, dyslexia, dyscalculia, dyspraxia,
    dysgraphia, neurodivergent/neurodiversity/neurotypical, sensory processing/overload, executive function,
    stimming, "on the spectrum", learning disability, developmental delay/disability, Tourette's, special needs,
    occupational/speech therapy, IEP, social skills.
  - **Education:** lesson plan, classroom, curriculum, pedagogy, teacher(s), educator(s), instructional,
    differentiated instruction, formative assessment, learning objectives/outcomes, scaffolding, homeschool,
    student engagement, literacy, special education, teaching strategies.
  - **Early childhood:** preschool(er), pre-K, kindergarten, toddler, early childhood/years, nursery school,
    phonics, alphabet, letter sounds, sight words, fine/gross motor, play-based, social-emotional, circle time,
    young children, daycare, early literacy, 3/4/5-year-olds, potty, nap time, storytime, counting to 10/20,
    colors and shapes.

## FineWeb-Edu: how much matches

| Domain | Loose (docs) | Strict (docs) | **Strict (tokens)** | In `sample-10BT` (10B) | In full FineWeb-Edu (1.3T) |
|---|---|---|---|---|---|
| Neurodivergence | 2.1% | 0.61% | **0.74%** | ~74M tokens | ~10B tokens |
| Education | 12.0% | 3.96% | **4.53%** | ~450M | ~59B |
| Early childhood | 4.9% | 0.81% | **0.86%** | ~86M | ~11B |

The shares were stable between a 40k-document and a 120k-document scan. Strict-matching documents overlap
somewhat: of the 120k, 289 matched early childhood + education, 130 education + neurodivergence, 23 early
childhood + neurodivergence, and 17 all three.

**Supply isn't the constraint; filtering cost is.** Collecting N tokens of a domain at share s means streaming
N/s tokens. 100M tokens of neurodivergence text means streaming ~14B tokens. This scan ran at ~0.4M tokens/s on
one CPU core, so that's ~9–10 CPU-hours: a CPU-only job (laptop or CPU Colab runtime) whose output is cached.
A trained classifier would make filtering faster and more precise later.

### What the matches look like

- **Neurodivergence.** Top terms: autism, ADHD, autistic, dyslexia, IEP, Asperger's, learning disabilities.
  Top sites include education.com, autismspeaks.org, myaspergerschild.com, everydayhealth.com, healthline.com.
  Mostly *about* neurodivergent people from a **medical, clinical or parent** perspective (diagnosis news,
  therapy services, parenting advice). Little neurodiversity-affirming or first-person writing, and some dated
  terminology and framing.
- **Education.** Top terms: teachers, classroom, curriculum, literacy, educators. Sites: edweek.org,
  teacherspayteachers.com, edutopia.org, scholastic.com, readwritethink.org. Mostly solid teacher-facing
  material; some noise (marketing, off-topic pages that mention teachers).
- **Early childhood.** Top terms: preschool, kindergarten, young children, early childhood, toddlers,
  alphabet, phonics. Sites: education.com, scholastic.com, brighthubeducation.com, notimeforflashcards.com,
  readingrockets.org, pbs.org. Mostly **adult-facing content about young children** (parenting, preschool
  policy, activity ideas, printables), not text written at a pre-K/K level.

## Cosmopedia: the young-children audience

| Subset (full rows) | `young_children` share | ≈ docs | Mean tokens* | ≈ tokens* |
|---|---|---|---|---|
| stories (4.99M) | 25.8% (format `story_children`) | ~1.29M | 591 | ~760M |
| stanford (1.02M) | 25.1% | ~256k | 1,035 | ~265M |
| openstax (126k) | 25.3% | ~32k | 869 | ~28M |
| auto_math_text (1.95M) | 50.2% `grade_school_students` | ~980k | 712 | ~695M |

\*Cosmopedia counts tokens with the Mistral-7B tokenizer.

Licence Apache-2.0. **What it is:** synthetic text in a child-friendly *voice* about mostly older-audience
topics: economics, energy investment, Japanese, poetry, climate, culture. It reads at roughly a 4th-grade style
and is formulaic ("Once upon a time, in a village called…", talking animals, "Mr. Wise Owl"). It's useful for
**simple, warm explanations and child-appropriate language**. It doesn't teach pre-K/K content (letters,
colors, counting, shapes, feelings, routines).

## Implications for the data mixture

1. **Domain filters** are keyword-density filters with the strict threshold above, run as a **CPU-only
   "filter and cache" job** over a large stream, writing domain shards to Drive. Training never waits on
   filtering.
2. **Cosmopedia filtering** by `audience` / `format` columns (e.g. `story_children`, `young_children`).
3. **Per-domain validation sets** built from held-out documents of each filtered slice. These measure "knows
   more than usual" directly.
4. **Epoch tracking per source.** Upweighting a small slice repeats it, and repeated text loses value after a
   few epochs (A8). The mixer reports each source's planned epochs and warns past ~4.
5. **Gaps that filtering can't fill:**
   - *Pre-K/K-level knowledge written for children* (if that's the goal, rather than knowledge about early
     childhood) would need synthetic generation: Cosmopedia-style, from a pre-K curriculum seed list, written by
     a strong model.
   - *Neurodiversity-affirming content* would need source curation (down-weighting deficit-focused sources)
     and possibly a curated or synthetic supplement.

## Decisions (2026-10-10)

- **Early childhood: knowledge *about* early childhood** (option b): development, teaching young children,
  parenting. FineWeb-Edu's filtered slice covers it; no paid synthetic generation.
- **Neurodivergence: neurodiversity-affirming**, because **children will talk to the model**. The
  `neurodivergence` source in `mixes/chatbot-v1.yaml` leaves out sites and phrasings with cure, deficit or
  vaccine-misinformation framing. Those lists are a first pass for review. A positive signal (preferring
  affirming, first-person and self-advocacy writing) would be the next refinement.
- Child-appropriate language comes partly from Cosmopedia's `story_children` stories, at a low weight.
