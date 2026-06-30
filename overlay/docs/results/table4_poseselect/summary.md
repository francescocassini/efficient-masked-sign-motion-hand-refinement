# P6-K Full R5 Aggregate

Replicas: **5**

| Dataset | Variant | PA-body | PA-hand | speed/GT | tstd/GT |
|---|---|---:|---:|---:|---:|
| csl | p5_cache | 8.3125 +/- 0.0199 | 1.8072 +/- 0.0012 | 0.772 +/- 0.001 | 0.611 +/- 0.000 |
| csl | p6d_clean | 8.2962 +/- 0.0200 | 1.6908 +/- 0.0018 | 0.734 +/- 0.001 | 0.586 +/- 0.002 |
| csl | p6k_clean | 8.2931 +/- 0.0200 | 1.6854 +/- 0.0017 | 0.767 +/- 0.001 | 0.634 +/- 0.001 |
| phoenix | p5_cache | 6.7732 +/- 0.0272 | 1.3519 +/- 0.0032 | 0.546 +/- 0.002 | 0.588 +/- 0.002 |
| phoenix | p6d_clean | 6.7662 +/- 0.0270 | 1.3127 +/- 0.0023 | 0.455 +/- 0.001 | 0.509 +/- 0.002 |
| phoenix | p6k_clean | 6.7631 +/- 0.0273 | 1.3127 +/- 0.0023 | 0.522 +/- 0.001 | 0.578 +/- 0.002 |

## Claim Rule

- This is a five-replica P6-K evaluation over separately saved final-fair cache replicas.
- P6-K remains a post-P5-cache method; it should be compared as an extension over the matched P5-cache rows.
- Insert into the SOKE/P3/P5 R5 table only with the explicit note that the P6-K base is the saved P5-cache wrapper path.
