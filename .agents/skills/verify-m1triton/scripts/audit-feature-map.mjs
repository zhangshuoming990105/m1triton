#!/usr/bin/env node
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const skillDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(skillDir, "../../..");
const featuresDir = join(skillDir, "references/features");
const REQUIRED_HEADINGS = ["## EntryPoints", "## Observable Behaviors", "## Invariants"];

function extractSection(text, heading) {
  const idx = text.indexOf(heading);
  if (idx === -1) return "";
  const after = text.slice(idx + heading.length);
  const next = after.search(/\r?\n##\s+/);
  return (next === -1 ? after : after.slice(0, next)).trim();
}

const files = readdirSync(featuresDir)
  .filter((f) => f.endsWith(".md") && f !== "README.md")
  .sort();

const errors = [];
if (files.length === 0) {
  errors.push("No Feature Map markdown files found under references/features/");
}

for (const file of files) {
  const full = join(featuresDir, file);
  const text = readFileSync(full, "utf8");
  for (const h of REQUIRED_HEADINGS) {
    const body = extractSection(text, h);
    if (!text.includes(h)) {
      errors.push(`${file}: missing required section "${h}"`);
    } else if (!body) {
      errors.push(`${file}: section "${h}" is empty`);
    }
  }
  const entrySection = extractSection(text, "## EntryPoints");
  const entryMatches = [...entrySection.matchAll(/`([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)`/g)];
  if (entryMatches.length === 0) {
    errors.push(`${file}: "## EntryPoints" must list at least one backticked repo file path`);
  }
  for (const m of entryMatches) {
    if (!existsSync(join(repoRoot, m[1]))) {
      errors.push(`${file}: EntryPoint "${m[1]}" does not exist in repository`);
    }
  }
}

if (errors.length > 0) {
  console.error("Feature Map audit FAILED:\n" + errors.map((e) => "  - " + e).join("\n"));
  process.exit(1);
}
console.log(`Feature Map audit OK (${files.length} file${files.length === 1 ? "" : "s"}).`);
