# English commentary translation audit

Only explanatory report/plot literal strings were translated. Existing comments
were already English. Source Chinese zone names, workbook sheet/column names,
and historical path literals remain data values. No control flow, computations,
model parameters, seeds, feature selection or data literals changed.

Each source file was tokenized: only STRING token payloads in the explicit
translation dictionary were replaced. Whitespace and all other tokens were
retained. The complete unified diff is `comment_translation.diff`; original and
translated SHA256 plus normalized-AST equality are in `comment_translation_audit.json`.
The equality check normalizes only the enumerated explanatory translations.
Corrected-script snapshots receive the same commentary-only treatment.
Historical Fig. 13 decision-map prose is labelled Fig. 11 in the translation.
Historical staging paths are retained for algorithm provenance; they are not
portable entry points. Use `zonerel` and current `paper/reproduce_*.py` adapters.

Files audited: 42. All normalized AST comparisons pass.

The before/after SHA-256 fields hash UTF-8 text with normalized line endings; final_file_bytes_sha256 hashes the actual final file bytes. Windows CRLF does not change the computational AST.
