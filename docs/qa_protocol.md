# QA Protocol

## Research QA
Reject if any of 12 criteria fail. Score 10 dimensions; accept >= 85/100.

## Final QA
22 checks with severity BLOCKER/MAJOR/MINOR/NIT. No release with unresolved
BLOCKER or MAJOR.

## Completed real-evaluation gate

The real OpenRouter evaluation gate has been satisfied through Kaggle
execution:

1. `OpenRouterRunner._real_call()` is implemented using the official
   OpenRouter Chat Completions endpoint with temperature 0, non-streaming
   requests, and budget guard approval before each call.
2. OpenRouter prices were verified on Kaggle in notebook 02 (21 eligible
   models, DeepSeek V4 Flash eligible at 0.09/0.18 USD per 1M tokens).
3. At least one model is eligible with a verified price.
4. The API key is loaded from Kaggle Secrets (never printed, logged, or
   written to disk).
5. The budget guard approves each call before it is made.
6. Notebook 04 completed on Kaggle: 20 real OpenRouter calls attempted, 15
   succeeded, 5 failed due to rate limit on a free model. Total token-based
   cost was US$0.000088, well below the US$0.50 smoke-run cap.
7. No local execution occurred at any point.
8. Notebook 05 final QA completed on Kaggle: 44 tests passed, 121 QA checks
   passed, 0 BLOCKER, 0 MAJOR.

Future larger evaluations still require budget approval and Kaggle execution.
The smoke-run results are tiny validation results and should not be interpreted
as a definitive model leaderboard.

## Execution
Run notebook 05_kaggle_final_qa.ipynb on Kaggle. Do not run QA locally.
