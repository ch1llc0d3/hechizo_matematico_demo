# AGENTS.md

Working conventions for AI agents in this repository. These rules are set by the human owner — follow them on every change.

## The collaboration flow (required)

1. **Never commit directly on `main`.** `main` is the owner's to merge.
2. **Create an isolated worktree per task** — never work in the main checkout:
   ```bash
   git worktree add ../hechizo_matematico_demo-<topic> -b feat/<topic>
   ```
3. **Make all changes inside the worktree.** The main checkout stays untouched.
4. **Commit** there with a clear, descriptive message.
5. **Push the branch and open a PR:**
   ```bash
   git push origin feat/<topic>   # if SSH push is denied, push an explicit HTTPS URL via gh's credential helper
   gh pr create --base main --head feat/<topic>
   ```
6. **Stop at the PR.** The owner reviews and merges manually. Agents never merge, never push to `main`, and never close their own PRs.
7. **After the owner merges:** optionally verify CI / the Pages deploy went green, then clean up:
   ```bash
   git worktree remove ../hechizo_matematico_demo-<topic>
   git branch -d feat/<topic>
   ```

## Repo quick facts

- Python demo "spells" — see `README.md` for what each script does and how to run them.
- The static GitHub Pages site lives in `docs/` (self-contained `index.html`, no build step, no external CDNs).
- Pages deploys automatically via `.github/workflows/deploy-pages.yml` when `docs/**` changes land on `main` (Pages source: GitHub Actions).
- Live site: https://ch1llc0d3.github.io/hechizo_matematico_demo/
- License: MIT.
