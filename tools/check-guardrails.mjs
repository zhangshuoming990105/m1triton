#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import { extname, join, posix, relative, resolve, sep } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const BUILTIN_ENFORCERS = new Set(["COMMENT-HYGIENE", "FROZEN-EVAL", "SKILL-PORTABILITY"]);
const SKIP_DIRS = new Set([".git", "node_modules", ".venv", "__pycache__", ".audit", "dist", "build"]);
const CODE_EXTS = new Set([
  ".js",
  ".mjs",
  ".cjs",
  ".ts",
  ".tsx",
  ".jsx",
  ".py",
  ".rs",
  ".go",
  ".c",
  ".cc",
  ".cpp",
  ".cxx",
  ".h",
  ".hpp",
  ".mm",
  ".metal",
]);
const C_FAMILY_EXTS = new Set([".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".mm", ".metal"]);

const ALLOW_TOKEN_RE =
  /guardrail-allow\(\s*([A-Z0-9_-]+)\s*,\s*owner=@([a-zA-Z0-9_.-]+)\s*,\s*expires=(\d{4}-\d{2}-\d{2})\s*\)\s*:\s*(\S.*)$/;

const RAW_SUPPRESSION_RE =
  /(?:@ts-ignore|@ts-expect-error|eslint-disable(?:-next-line|-line)?|#\s*type:\s*ignore|#\s*noqa\b|#\[allow\()/;

const NARRATION_COMMENT_RE =
  /^\s*(?:\/\/|#)\s*(?:Phase\s+\d+\s*:|Step\s+\d+\s*:|Added\s+for\b|Refactored\s+from\b|Changed\s+from\b)/i;

const WORKAROUND_COMMENT_RE =
  /^\s*(?:\/\/|#)\s*(?:TODO:\s*hack\b|FIXME:\s*workaround\b|HACK:|temporary\s+workaround\b)/i;

const toPosix = (p) => p.split(sep).join("/");

function walkCodeFiles(root, current = root, out = []) {
  for (const entry of readdirSync(current, { withFileTypes: true })) {
    if (SKIP_DIRS.has(entry.name)) continue;
    const full = join(current, entry.name);
    if (entry.isDirectory()) {
      walkCodeFiles(root, full, out);
    } else if (entry.isFile() && CODE_EXTS.has(extname(entry.name))) {
      out.push(full);
    }
  }
  return out.sort();
}

function sha256File(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function parseAllowance(line, today) {
  const commentMatch = line.match(/(?:\/\/|#)\s*(.*guardrail-allow\(.*)$/);
  if (!commentMatch) return null;
  const match = commentMatch[1].match(ALLOW_TOKEN_RE);
  if (!match) {
    return {
      malformed: true,
      reason:
        "Malformed guardrail-allow token. Expected: guardrail-allow(RULE_ID, owner=@handle, expires=YYYY-MM-DD): reason",
    };
  }
  const [, ruleId, owner, expires, reason] = match;
  if (expires < today) {
    return {
      expired: true,
      ruleId,
      owner,
      expires,
      reason: `Allowance for ${ruleId} (owner=@${owner}) expired on ${expires}. Fix the root cause or renew with human review.`,
    };
  }
  return { ruleId, owner, expires, reason };
}

function resolvePythonModule(rawSpec, relFile) {
  const dots = rawSpec.match(/^\.+/)?.[0].length ?? 0;
  if (dots === 0) return rawSpec.replaceAll(".", "/");
  const rest = rawSpec.slice(dots).replaceAll(".", "/");
  let base = posix.dirname(relFile);
  for (let i = 1; i < dots; i++) base = posix.dirname(base);
  return posix.normalize(rest ? `${base}/${rest}` : base);
}

function resolveJsSpecifier(rawSpec, relFile) {
  if (!rawSpec.startsWith(".")) return rawSpec;
  return posix.normalize(posix.join(posix.dirname(relFile), rawSpec));
}

function extractImports(line, ext, relFile) {
  const specs = [];
  if (ext === ".py") {
    const trimmed = line.trim();
    if (trimmed.startsWith("#")) return specs;
    const fromMatch = trimmed.match(/^from\s+([.\w]+)\s+import\s+/);
    if (fromMatch) specs.push(resolvePythonModule(fromMatch[1], relFile));
    const impMatch = trimmed.match(/^import\s+([\w.,\s]+)/);
    if (impMatch) {
      for (const part of impMatch[1].split(",")) {
        const token = part.trim().split(/\s+as\s+/)[0].trim();
        if (token) specs.push(token.replaceAll(".", "/"));
      }
    }
  } else if ([".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"].includes(ext)) {
    const trimmed = line.trim();
    if (trimmed.startsWith("//") || trimmed.startsWith("*")) return specs;
    for (const m of line.matchAll(/(?:from\s+|import\s*\(\s*|require\s*\(\s*)["']([^"']+)["']/g)) {
      specs.push(resolveJsSpecifier(m[1], relFile));
    }
  } else if (C_FAMILY_EXTS.has(ext)) {
    const trimmed = line.trim();
    if (trimmed.startsWith("//") || trimmed.startsWith("*")) return specs;
    const incMatch = trimmed.match(/^#\s*include\s*["<]([^">]+)[">]/);
    if (incMatch) {
      const rawInc = incMatch[1];
      if (rawInc.startsWith(".")) {
        specs.push(posix.normalize(posix.join(posix.dirname(relFile), rawInc)));
      } else {
        specs.push(rawInc);
      }
    }
  }
  return specs;
}

function matchesPrefix(relPath, prefixes) {
  if (!prefixes || prefixes.length === 0) return true;
  return prefixes.some((p) => relPath.startsWith(p) || relPath.endsWith(p));
}

function validateConfigSchema(config) {
  const errors = [];
  for (const b of config.boundaries ?? []) {
    if (!b.id || !b.from || !Array.isArray(b.forbidImportsFrom) || !b.fix?.trim()) {
      errors.push(
        `[SCHEMA-MISSING-FIX] guardrails.json boundary "${b.id ?? "<unnamed>"}": must define id, from, forbidImportsFrom[], and non-empty "fix".`
      );
    }
  }
  for (const p of config.bannedPatterns ?? []) {
    if (!p.id || !p.pattern || !p.fix?.trim()) {
      errors.push(
        `[SCHEMA-MISSING-FIX] guardrails.json bannedPattern "${p.id ?? "<unnamed>"}": must define id, pattern, and non-empty "fix".`
      );
    }
  }
  return errors;
}

function checkAgentsLedger(root, config, configuredRuleIds) {
  const errors = [];
  const agentsPath = join(root, "AGENTS.md");
  if (!existsSync(agentsPath)) return errors;
  const text = readFileSync(agentsPath, "utf8");
  const sectionMatch = text.match(/## Rule Ledger[\s\S]*?(?=\n## |\s*$)/);
  if (!sectionMatch) return errors;

  const rows = sectionMatch[0]
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("|") && !/^\|[\s:-]+\|/.test(l));
  if (rows.length < 2) return errors;

  const ledgerRuleIds = new Set();
  let judgmentCount = 0;
  const maxJudgment = config.maxJudgmentOnlyRules ?? 3;

  for (const row of rows.slice(1)) {
    const cells = row
      .split("|")
      .slice(1, -1)
      .map((c) => c.trim().replace(/^`(.*)`$/, "$1"));
    if (cells.length < 4) continue;
    const [ruleId, layer, enforcer] = cells;
    ledgerRuleIds.add(ruleId);

    if (enforcer.startsWith("guardrail:")) {
      const targetId = enforcer.slice("guardrail:".length).trim();
      if (!configuredRuleIds.has(targetId) && !BUILTIN_ENFORCERS.has(targetId)) {
        errors.push(
          `[UNENFORCED-RULE] AGENTS.md rule "${ruleId}" references "${enforcer}", missing from guardrails.json.`
        );
      }
    } else if (enforcer.startsWith("script:") || enforcer.startsWith("test:")) {
      const relTarget = enforcer.slice(enforcer.indexOf(":") + 1).trim();
      const absTarget = join(root, relTarget);
      if (!existsSync(absTarget) || !statSync(absTarget).isFile()) {
        errors.push(`[UNENFORCED-RULE] AGENTS.md rule "${ruleId}" points to missing file "${relTarget}".`);
      }
    } else if (enforcer.startsWith("judgment-only:")) {
      const reason = enforcer.slice("judgment-only:".length).trim();
      if (layer !== "docs" || reason.length < 8) {
        errors.push(
          `[INVALID-JUDGMENT-RULE] AGENTS.md rule "${ruleId}": judgment-only rules require Layer="docs" and a concrete reason.`
        );
      }
      judgmentCount++;
    } else {
      errors.push(
        `[PROSE-ONLY-RULE] AGENTS.md rule "${ruleId}" has invalid Enforcer "${enforcer}". Prose without an enforcer is banned.`
      );
    }
  }

  if (judgmentCount > maxJudgment) {
    errors.push(
      `[TOO-MANY-PROSE-RULES] AGENTS.md has ${judgmentCount} judgment-only rules (max ${maxJudgment}).`
    );
  }
  for (const id of configuredRuleIds) {
    if (!ledgerRuleIds.has(id)) {
      errors.push(`[LEDGER-DRIFT] guardrails.json rule "${id}" is missing from AGENTS.md "## Rule Ledger".`);
    }
  }
  return errors;
}

export function runGuardrails(root, options = {}) {
  const today = options.today ?? process.env.GUARDRAIL_TODAY ?? new Date().toISOString().slice(0, 10);
  const configPath = join(root, "guardrails.json");
  if (!existsSync(configPath)) {
    return [`[CONFIG-MISSING] guardrails.json not found in ${root}`];
  }

  const config = JSON.parse(readFileSync(configPath, "utf8"));
  const violations = [...validateConfigSchema(config)];
  if (violations.length > 0) return violations;

  const configuredRuleIds = new Set([
    ...(config.boundaries ?? []).map((b) => b.id),
    ...(config.bannedPatterns ?? []).map((p) => p.id),
  ]);

  violations.push(...checkAgentsLedger(root, config, configuredRuleIds));

  for (const [relFrozen, expectedHash] of Object.entries(config.frozenFiles ?? {})) {
    const absFrozen = join(root, relFrozen);
    if (!existsSync(absFrozen)) {
      violations.push(
        `[FROZEN-EVAL] ${relFrozen}: missing frozen file. Fix: restore ${relFrozen} from git.`
      );
      continue;
    }
    const actualHash = sha256File(absFrozen);
    if (actualHash !== expectedHash) {
      violations.push(
        `[FROZEN-EVAL] ${relFrozen}: SHA-256 mismatch (expected ${expectedHash.slice(0, 12)}..., got ${actualHash.slice(0, 12)}...). Fix: git checkout -- ${relFrozen}.`
      );
    }
  }

  const compiledPatterns = (config.bannedPatterns ?? []).map((p) => ({
    ...p,
    regex: new RegExp(p.pattern),
  }));
  const hygieneEnabled = config.commentHygiene?.enabled !== false;
  const hygieneIgnorePrefixes = config.commentHygiene?.excludePrefixes ?? ["tests/", "tools/", ".agents/"];

  for (const absFile of walkCodeFiles(root)) {
    const relFile = toPosix(relative(root, absFile));
    const ext = extname(absFile);
    const lines = readFileSync(absFile, "utf8").split("\n");
    const skipHygiene = !hygieneEnabled || matchesPrefix(relFile, hygieneIgnorePrefixes);

    lines.forEach((line, idx) => {
      const lineNo = idx + 1;
      const allowance = skipHygiene ? null : parseAllowance(line, today);
      if (allowance?.malformed) {
        violations.push(`[COMMENT-HYGIENE] ${relFile}:${lineNo}: ${allowance.reason}`);
        return;
      }
      if (allowance?.expired) {
        violations.push(`[EXPIRED-ALLOWANCE] ${relFile}:${lineNo}: ${allowance.reason}`);
        return;
      }

      const imports = extractImports(line, ext, relFile);
      for (const boundary of config.boundaries ?? []) {
        if (!relFile.startsWith(boundary.from)) continue;
        if (allowance?.ruleId === boundary.id) continue;
        for (const spec of imports) {
          const hit = boundary.forbidImportsFrom.find(
            (forbid) => spec === forbid || spec.startsWith(`${forbid}/`) || spec.startsWith(`${forbid}.`)
          );
          if (hit) {
            violations.push(
              `[${boundary.id}] ${relFile}:${lineNo}: "${relFile}" imports forbidden target "${spec}" (matches "${hit}"). Fix: ${boundary.fix}`
            );
          }
        }
      }

      for (const pat of compiledPatterns) {
        if (pat.includePrefixes && !matchesPrefix(relFile, pat.includePrefixes)) continue;
        if (pat.excludePrefixes && matchesPrefix(relFile, pat.excludePrefixes)) continue;
        if (allowance?.ruleId === pat.id) continue;
        if (pat.regex.test(line)) {
          violations.push(
            `[${pat.id}] ${relFile}:${lineNo}: banned pattern matched (${pat.pattern}). Fix: ${pat.fix}`
          );
        }
      }

      if (!skipHygiene && allowance?.ruleId !== "COMMENT-HYGIENE") {
        if (RAW_SUPPRESSION_RE.test(line)) {
          violations.push(
            `[COMMENT-HYGIENE] ${relFile}:${lineNo}: unapproved lint/type suppression. Fix: solve the root issue or add guardrail-allow(COMMENT-HYGIENE, owner=@you, expires=YYYY-MM-DD): <why>`
          );
        } else if (NARRATION_COMMENT_RE.test(line)) {
          violations.push(
            `[COMMENT-HYGIENE] ${relFile}:${lineNo}: phase/changelog narration comment detected. Fix: delete it; leave history to git log.`
          );
        } else if (WORKAROUND_COMMENT_RE.test(line)) {
          violations.push(
            `[COMMENT-HYGIENE] ${relFile}:${lineNo}: unstructured hack/workaround comment. Fix: resolve root cause or add expiring guardrail-allow.`
          );
        }
      }
    });
  }

  return violations;
}

const isMain =
  Boolean(process.argv[1]) &&
  realpathSync(resolve(process.argv[1])) === realpathSync(fileURLToPath(import.meta.url));
if (isMain) {
  const targetRoot = resolve(process.argv[2] ?? process.cwd());
  const violations = runGuardrails(targetRoot);
  if (violations.length > 0) {
    console.error(`Guardrail check FAILED (${violations.length} violation${violations.length === 1 ? "" : "s"}):\n`);
    for (const v of violations) console.error(`  - ${v}`);
    process.exit(1);
  }
  console.log("Guardrails OK.");
}
