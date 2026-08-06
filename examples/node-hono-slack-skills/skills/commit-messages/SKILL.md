---
name: commit-messages
description: Write a Conventional Commits message from a diff or change summary. Use when the user asks for a commit message or mentions committing.
---

# Commit messages

Write a Conventional Commits subject, then an optional body.

1. Pick a type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
2. Subject: `type(scope): summary` — imperative, lowercase, no trailing period, <= 72 chars.
3. Body (optional): what changed and why, wrapped at 72 columns.
4. Note breaking changes in a `BREAKING CHANGE:` footer.
