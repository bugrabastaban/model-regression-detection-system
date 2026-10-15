# Prompt evaluation CI

This workflow runs on PRs targeting `main`. A stable `Prompt evaluation gate`
check exists even when `prompts/` is unchanged; those PRs use no live quota.
The separate `Offline tests` job runs on every PR.

## Initial installation

1. Commit/push this workflow, `ci_pipeline.py`, `ci/`, tests and `.gitignore`
   to **main first**. The live job checks out trusted base-branch code, so its
   scripts must already exist on the target branch before testing prompt PRs.
2. In repository **Settings → Secrets and variables → Actions**, add:
   `GEMINI_API_KEY`, `GROQ_API_KEY`; `SLACK_WEBHOOK_URL` is optional.
   Never commit `.env`. Rotate a webhook if it was shared outside its intended
   secret storage, and update the repository secret and local environment.
3. Open a same-repository PR adding `prompts/v2.yaml` (with `version_id: v2`),
   or modifying an existing YAML prompt. The highest numbered `vN.yaml` in
   the PR tree is evaluated. Every versioned prompt in the tree is validated.
   Editing an older inactive version still triggers evaluation of the highest
   version; it does not silently switch the selected prompt backwards.
4. After the first workflow check appears, configure a ruleset/branch protection
   for `main` requiring **Offline tests** and **Prompt evaluation gate**.
   Enable required PRs and restrict bypass as appropriate. A failed workflow
   alone does not prevent merges without this repository setting. Availability
   of protection for private repositories depends on the GitHub plan.
5. Verify the PR comment, HTML/JSON artifact and optional Slack notification.
   Artifact links require GitHub access, download results rather than host a
   public HTML site, and expire after 30 days.

This implementation has been tested locally with mocks; it has not been run
on GitHub Actions or against providers as part of its installation.

## Scope and baseline

Exactly `email_002`, `email_001`, `email_091` are run, serially. No full-dataset
evaluation is launched. Each PR uses at most **3 Gemini + 3 Groq** requests.
Each completed case is followed by a wait of at least 60 seconds before the
next Gemini request. Provider/validation failures stop the run, without retry.
The installed SDK retry policy is checked before live calls. A job timeout
does not guarantee an already running request was not charged.

`ci/config.json` selects the cohort, models and approved baseline paths.
`ci/baseline.json` preserves the three successful live results consolidated
on 2026-10-15; it is a fixed baseline, not a fresh run or drift history entry.
`ci/baseline_dataset.json` preserves the three reference cases and
`ci/baseline_prompts/v1.yaml` preserves the baseline prompt independently of
later edits to `prompts/v1.yaml`. The stored prompt digest guards accidental
baseline configuration changes. Baseline promotion remains an explicit human
review; PR results never overwrite it automatically.

The trusted base branch supplies model-calling/scoring/comparison/report code,
dependencies, config, dataset and baseline. Only regular `prompts/vN.yaml`
blobs are extracted from the PR commit. PR Python files are not executed with
provider secrets. Fork prompt PRs are marked INCOMPLETE and make no live calls;
they do not gain secret access through `pull_request_target`.

## Quota budget

Default CI allowance: **3 reserved Gemini calls per rolling 24 hours** across
this repository. A reservation artifact is uploaded **before** live calls;
failures/cancellations still consume all 3 reserved calls. Jobs serialize under
one concurrency group. A rerun requires a new reservation, so it does not bypass
the allowance. Budget exhaustion is INCOMPLETE, not a successful evaluation.

To change the allowance, set repository **variable** `GEMINI_CI_DAILY_BUDGET`
to an integer from 0 to 20 (0 disables live requests). Do not set 20 unless all
other uses of that key/project have been accounted for. Reservations measure
this workflow only: manual calls, other repositories and other services are
not visible. Keep sufficient provider quota available before opening a prompt
PR. Never delete reservation artifacts to force another run; they are retained
for 7 days. This conservative rolling window is not a provider quota API and
does not assume a particular provider reset timezone.

GitHub concurrency allows only a limited pending queue; rapid PR updates can
replace a pending run. Superseded/cancelled checks must not be treated as a pass.

## Outcomes

- PASS: complete paired scores without a warning/critical threshold.
- WARN: complete results with warning-level changes; does not block merge.
- FAIL: critical regression (>8 percentage points by default, including a
  category-level critical change); merge check fails.
- INCOMPLETE: provider/validation/config/quota/fork failure, missing paired
  results, or failed offline tests; merge check fails.
- NOT_APPLICABLE: no prompt changes; live calls skipped, check succeeds.

For a 3-case cohort a single pass→fail change is a large percentage change;
these thresholds should not be interpreted as statistical significance.
History/drift aggregation is not automatically created by this workflow.
Reports retain the current dataset snapshot for future comparisons.

The PR summary comment is updated in place. It includes pass-rate delta,
regression/improvement counts and the artifact link. Provider errors in uploaded
JSON are sanitized; no raw stack traces or provider error bodies are published.
The artifact uploads only explicit output files, never `.env` or the workspace.

When the optional Slack secret exists, a single notification is sent with the
same artifact link. Failed delivery is logged without secrets and is not
retried; it cannot override the evaluation gate. No notification is sent by
importing modules or by running offline tests.

## Local offline validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Do not run `python ci_pipeline.py evaluate` locally just to test the workflow:
that command intentionally uses real configured providers. It requires a
prepared plan; the GitHub workflow reserves quota before invoking it.
