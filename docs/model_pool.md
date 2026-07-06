# Model Pool

## Eligibility
- input_price_per_million < 0.12 USD
- output_price_per_million < 0.30 USD
- price verified with timestamp

Default eligible: false. See configs/model_pool.yaml.

## Baseline reference
DeepSeek V4 Flash is the cheap baseline reference, not a target to always beat.

## Verification
Run notebook 02_kaggle_openrouter_price_check.ipynb on Kaggle to verify and
update eligibility.
