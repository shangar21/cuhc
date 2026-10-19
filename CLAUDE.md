# cuhc project instructions

## Documentation standard: ASD-STE100 (Simplified Technical English)

Write all new and changed documentation in ASD-STE100. This applies to `README.md`,
`docs/*.md` and user-facing docstrings. (The user asked for "ESD-STE1000"; we read it as
ASD-STE100. Ask the user if this is not correct.) Code comments can stay in plain
technical English.

Each document must stand alone:

- Do not mention the writing standard (ASD-STE100) in any document.
- Write only about the topic of the document: how the code operates, how to use it and
  why it has this design. Describe the current state.
- Do not write the history of the work (for example "the first version", "this step
  replaced", "a question during step 4", "update:"). Measured comparisons between two
  methods are permitted when they explain a design decision.

The main rules:

- Use the STE approved words, with their approved meanings only. Technical names (for
  example `linkage`, `centroid`, `kernel`, `GPU`) and technical verbs for computer
  processes (for example `compile`, `merge`, `return`) are permitted.
- Do not use words that are not approved. Examples, with an approved alternative:
  "get" (use "receive" or "find"), "need" (use "is necessary" or "must"), "like"
  (use "the same as" or "for example"), "about" for a quantity (use "approximately"),
  "allow" (use "let" or "permit"), "ensure" (use "make sure").
- A procedural sentence (an instruction) has a maximum of 20 words. A descriptive
  sentence has a maximum of 25 words.
- Write one instruction in each sentence. Use the imperative ("Build the library.").
- Use the active voice in procedures. Use the passive voice in descriptions only if
  necessary.
- Use only the simple tenses: present, past and future.
- Do not use "-ing" forms, except in technical names.
- Do not use contractions ("does not", not "doesn't").
- Do not leave out words to make a sentence shorter. Keep articles ("the", "a") and verbs.
- Do not put more than three nouns together in a noun cluster.
- A paragraph has a maximum of six sentences and one topic.
- Put a condition before the instruction ("If X occurs, do Y.").
- Use vertical lists and tables for items and steps. Number the steps of a procedure.
- Write warnings and cautions as commands, and give the risk and the result.

## Build and test

- Build: `cmake -S . -B build && cmake --build build -j` (writes `python/cuhc/libcuhc.so`).
- Test: `python3 -m pytest tests -q -p no:cacheprovider`.
- Benchmarks: `bench/bench_linkage.py`, `bench/bench_hdbscan.py`, `./build/bench_distance`.
- Docs site: `pip install -r docs/requirements.txt && mkdocs build` (MkDocs Material; API pages from docstrings via mkdocstrings, which reads the source without importing it).
