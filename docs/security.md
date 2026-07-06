# Security

- No secrets in files, commits, or logs.
- Load API keys from Kaggle Secrets only.
- Never create a .env with real keys.
- .gitignore excludes secret-bearing patterns.
- No local API calls.
- No uncontrolled spending (budget guard enforces caps).
- No model endpoint without price verification.
- No hidden local dependency.
