# Portfolio consolidation plan (drafted 2026-11-09, for the 16 November deadline)

**Goal.** Make the confirmed Checkpoint 2.4 study the obvious entry point on GitHub. A reviewer should understand the finding, its status and how to reproduce it within two minutes, without navigating checkpoint branches.

**Current state.**
- `main` holds only the initial commit, with a one-line README.
- All 109+ commits live on `checkpoint-2.3/exploratory-paths` (and older checkpoint branches).
- A visitor to the repository today sees an empty project.

## Prepared on the working branch (no approval needed; done)

| Item | File | Purpose |
|---|---|---|
| Research README | `README.md` | Figure 1, the three confirmed results, a claim-status table (confirmed / structural / exploratory / in progress / not established), reproduction, and project history including the stopped H\* |
| Preprint-style manuscript | `docs/paper/MANUSCRIPT.md` | A standalone paper: abstract, related work, method, development and held-out results, limitations, and pre-registration and reproducibility statements |
| Contribution summary | `docs/RESEARCH_SUMMARY.md` | Paste-ready 120-word and 300-word versions for applications |
| Wording audit | 2.4 `REPORT.md` | Claims narrowed to the two tested clean-gradient rules (magnitude, first-order). A dated note was added. |

## Steps that need your approval (not done)

Steps 1–2 are public, and they are what makes the work visible.

1. **Open a PR from `checkpoint-2.3/exploratory-paths` into `main`.**
   - `main` has only the initial commit, so this is a fast-forward-style merge with no conflicts expected.
   - The PR description would summarize the confirmed result and link the frozen protocol.
   - **Merging needs your explicit approval.** Opening the PR is also a public action, so I will not open it without your go.
   - Alternative if you prefer not to merge history: make the working branch the default branch on GitHub (Settings → Branches). This is a repository-settings change, so it is yours to make.
2. **Tag the confirmed state.** `v1.0-confirmed` at the commit that adds the report (`bd15a97`) or at the README commit. A tag gives reviewers and the manuscript a stable reference, independent of later Checkpoint 3 work. Pushing a tag is public, so it needs your approval.
3. **Repository metadata (optional, your action).** A one-line description ("Pre-registered evidence that PGD's advantage over FGSM on a VLM is coordinate selection") and topics: adversarial-robustness, vision-language-models, ai-safety, pre-registration.
4. **PDF of the manuscript (optional).** Render `MANUSCRIPT.md` to PDF for application portals that want a file. This is local only, so I can do it on request.
5. **Archive DOI (optional, after the deadline).** A Zenodo release of the tag. It needs your account.

## What stays as it is

- Checkpoint folders keep their frozen protocols, records and dated annotations. Nothing is rewritten for presentation.
- The README links into them rather than replacing them.
- Checkpoint 3 stays labeled "in progress / exploratory" in the README until a confirmation exists. If its pilot is reported before the deadline, it appears as a clearly marked exploratory section, never merged into the confirmed claim.

## Order of work before 16 November

1. Done today: README, manuscript, summary, wording audit, pushed on the working branch.
2. Your review of the README and manuscript text.
3. With approval: PR to `main`, merge, tag.
4. Only if time remains: Checkpoint 3 pilot write-up as an exploratory appendix.
