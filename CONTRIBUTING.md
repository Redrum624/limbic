# Contributing

## Dev setup

Node ≥ 22. Then:

```sh
npm ci               # install; the prepare script also builds dist/
npm test             # vitest — must be green, and makes zero network calls
npm run typecheck    # tsc --noEmit — must be silent
npm run build        # tsup — ESM + CJS + types
```

The live Ollama suite is opt-in (`LIMBIC_LIVE=1`) and is never required for a
PR; CI does not run it.

## The golden-fixture rule

`test/fixtures/*.json` are **pinned, imported artifacts** — copied from their
upstream repositories, sha256-pinned in `test/fixtures.hash.test.ts`, and
re-hashed on every run. Never hand-edit them, never regenerate them here, and
never "fix" a failing hash by re-pinning: a hash mismatch means the bytes
changed, and the bytes changing is the defect. Provenance for both fixtures,
including the one documented metadata exception, is in
`test/fixtures/PROVENANCE.md`. If an upstream fixture legitimately changes,
re-copy it from upstream and update `PROVENANCE.md` and the pinned hash in the
same commit, with the upstream reference stated.

## Pull requests

- Behavioural changes come with a test that fails before the change and passes
  after it. Mechanical rewording does not need one.
- `npm test`, `npm run typecheck` and `npm run build` all green locally before
  you push; CI runs the same three on Node 20 and 22.
- Don't introduce runtime dependencies. New integrations follow the existing
  pattern: an optional peer behind a dynamic `import()` with an install hint.
- Keep files LF-ended and BOM-free (`.gitattributes` enforces the fixtures).
- Constants need an argued source — a comment saying where the value comes from,
  not a bare number.
