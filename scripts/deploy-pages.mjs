#!/usr/bin/env node
/**
 * Publish the built page to GitHub Pages.
 *
 * GitHub Pages can be served two ways: by an Actions workflow (which needs
 * Actions to be enabled on the account) or straight from a branch. This script
 * uses the branch: it builds the static site, then commits `dist/` as the root
 * of a `gh-pages` branch and pushes it. No CI, no runner, nothing to bill.
 *
 *   npm run deploy
 *
 * One-time setup on the repository: Settings → Pages → Source: "Deploy from a
 * branch", Branch: `gh-pages`, folder `/ (root)`.
 */
import { execFileSync, execSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const run = (cmd, cwd, stdio = 'inherit') => execSync(cmd, { cwd, stdio, shell: true });

console.log('› building');
run('npm run build', root);

if (!existsSync(join(dist, 'index.html'))) {
  console.error('✗ dist/index.html is missing — the build did not produce a site');
  process.exit(1);
}

// Pages runs Jekyll over the branch by default; this opts out so that files and
// folders beginning with an underscore are served as-is.
writeFileSync(join(dist, '.nojekyll'), '');

const remote = execFileSync('git', ['remote', 'get-url', 'origin'], { cwd: root }).toString().trim();
const stamp = new Date().toISOString();

console.log(`› publishing dist/ → gh-pages on ${remote}`);
if (!existsSync(join(dist, '.git'))) run('git init -b gh-pages', dist);
run(`git add -A`, dist);
run(`git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "deploy: ${stamp}" --allow-empty`, dist);
run(`git remote remove origin 2>/dev/null || true`, dist);
run(`git remote add origin ${remote}`, dist);
run('git push -f origin gh-pages', dist);

console.log('✓ pushed. Pages URL: https://<user>.github.io/<repo>/');
