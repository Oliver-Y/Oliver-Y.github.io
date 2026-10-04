# Oliver-Y.github.io

Personal site: Eleventy 3, input `src/`, output `_site/`. Deployed to GitHub Pages by
`.github/workflows/` on push to `main` only — feature branches never deploy.

- Build/preview: `npm install && npx @11ty/eleventy --serve`
- Notes live in `src/notes/*.md` (front matter: `title`, `date`, `blurb`). There is no draft
  flag: anything in `src/notes/` is published as soon as it reaches `main`.
- Math is KaTeX via `@vscode/markdown-it-katex` (`$...$` inline, `$$...$$` block).
- This repo is **public**. Keep personal learning notes, work references, and anything
  private out of it — `.claude/` included.
- Commit identity is repo-local (`Oliver-Y <Oliver-Y@users.noreply.github.com>`). Never commit
  with a work email.

## In-progress work

- `oye_add_stats-notes`: the "Stats review" post. Read
  `.claude/checkpoints/checkpoint-stats-review-post.md` before working on it.
