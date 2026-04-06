# SpecEE Predictor Comparison

- Date: 2026-04-03 18:15
- Device: mps
- Threshold: 0.5
- Base model: meta-llama/Llama-2-7b-chat-hf
- Draft model: yuhuili/EAGLE-llama2-chat-7B
- Samples per dataset: 10

## Speed (tokens/s)

| Predictor | mt_bench | alpaca | gsm8k | qa | humaneval |
|---|---|---|---|---|---|
| HF Baseline | 14.09 | 15.12 | 14.17 | 15.35 | 13.44 |
| specee | 14.34 | 15.22 | 14.67 | 15.06 | 14.46 |
| jay-ap1 | 14.90 | 15.53 | 14.77 | 15.00 | 14.29 |
| jay-ap2 | 14.91 | 15.46 | 14.92 | 14.87 | 14.29 |
| wahab-ap1 | 15.00 | 16.07 | 14.90 | 14.86 | 14.22 |
| wahab-ap3 | 15.13 | 15.29 | 14.85 | 14.53 | 14.40 |

## Speedup (vs HF Baseline)

| Predictor | mt_bench | alpaca | gsm8k | qa | humaneval |
|---|---|---|---|---|---|
| HF Baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| specee | 1.02x | 1.01x | 1.04x | 0.98x | 1.08x |
| jay-ap1 | 1.06x | 1.03x | 1.04x | 0.98x | 1.06x |
| jay-ap2 | 1.06x | 1.02x | 1.05x | 0.97x | 1.06x |
| wahab-ap1 | 1.06x | 1.06x | 1.05x | 0.97x | 1.06x |
| wahab-ap3 | 1.07x | 1.01x | 1.05x | 0.95x | 1.07x |

## Accuracy

| Predictor | commonsenseqa | sst2 |
|---|---|---|
| HF Baseline | 60.00% | 90.00% |
| specee | 59.74% | 93.75% |
| jay-ap1 | 60.76% | 91.25% |
| jay-ap2 | 60.00% | 91.25% |
| wahab-ap1 | 60.00% | 93.75% |
| wahab-ap3 | 60.76% | 93.75% |

## Avg Exit Layer

| Predictor | mt_bench | alpaca | gsm8k | qa | humaneval | commonsenseqa | sst2 |
|---|---|---|---|---|---|---|---|
| specee | 24.6 | 24.7 | 24.9 | 25.2 | 24.5 | 26.1 | 28.2 |
| jay-ap1 | 25.4 | 24.9 | 25.2 | 25.7 | 25.5 | 26.2 | 29.1 |
| jay-ap2 | 25.4 | 24.8 | 25.1 | 25.9 | 25.6 | 26.3 | 29.0 |
| wahab-ap1 | 25.2 | 24.0 | 24.9 | 25.5 | 25.1 | 26.2 | 29.1 |
| wahab-ap3 | 24.7 | 24.6 | 24.7 | 26.0 | 24.8 | 26.1 | 29.1 |

