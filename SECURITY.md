# Security and data protection

## Scope

This is research code. It is **not a medical device**, has no clinical
validation, and must not be used to inform diagnosis or treatment.

## What this repository must never contain

- Ultrasound images, or any derivative from which an image could be reconstructed
- Patient identifiers, examination dates, accession numbers, or institutional
  file paths
- Model checkpoints trained on patient images

All study data, including anonymized tables and aggregate image features, must
remain in `data/` or `outputs/`; both directories are excluded by `.gitignore`.

Only synthetic input examples belong in `examples/`.

## Reporting a problem

If you believe this repository exposes patient data or contains a security
issue, report it privately through GitHub Security Advisories rather than
opening a public issue.
