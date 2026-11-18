Model: `models/classifier_hybrid.pkl` (in-process, CPU). 200 sequential writes: p50 392 ms, p95 655 ms, p99 910 ms, 2.5 req/s.
Concurrency: 4 threads x 25 requests in 96.5 s = 1.0 req/s, 0 server errors.
Robustness: 25/25 controlled responses; 5xx anywhere: False.

| Case | HTTP | Pass | Verdict |
|---|---|---|---|
| zero_width_split | 200 | yes | malicious |
| rtl_override | 200 | yes | malicious |
| homoglyphs | 200 | yes | malicious |
| null_bytes | 200 | yes | ordinary_learning |
| emoji_only | 200 | yes | malicious |
| very_long_note | 422 | yes |  |
| repeated_attack | 422 | yes |  |
| mixed_scripts | 200 | yes | malicious |
| control_chars | 200 | yes | malicious |
| only_punctuation | 200 | yes | malicious |
| single_char | 200 | yes | ordinary_learning |
| newlines_tabs | 200 | yes | malicious |
| html_script | 200 | yes | malicious |
| sql_like | 200 | yes | ordinary_learning |
| format_string | 200 | yes | malicious |
| malformed_json | 422 | yes |  |
| wrong_types | 422 | yes |  |
| missing_fields | 422 | yes |  |
| no_key | 401 | yes |  |
| wrong_key | 401 | yes |  |
| bad_origin | 422 | yes |  |
| bad_session_id | 422 | yes |  |
| oversized_body | 413 | yes |  |
| unknown_route | 404 | yes |  |
| docs_disabled | 404 | yes |  |
