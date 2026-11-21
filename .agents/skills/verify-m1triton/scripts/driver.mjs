#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import process from "node:process";

const root = resolve(process.cwd());
const hash = createHash("sha256").update(root).digest("hex").slice(0, 8);
const isolatedPort = 14000 + (Number.parseInt(hash.slice(0, 4), 16) % 5000);
const stateDir = join(root, ".audit", "verify-m1triton-" + hash);
const smokeCmd = "bash scripts/ci_local.sh";

const subcmd = process.argv[2] ?? "doctor";

if (subcmd === "doctor") {
  mkdirSync(stateDir, { recursive: true });
  console.log(JSON.stringify({ ok: true, root, isolatedPort, stateDir, smokeCmd }, null, 2));
} else if (subcmd === "drive" || subcmd === "evidence") {
  mkdirSync(stateDir, { recursive: true });
  const res = spawnSync(smokeCmd, {
    cwd: root,
    shell: true,
    encoding: "utf8",
    env: {
      ...process.env,
      VERIFY_PORT: String(isolatedPort),
      VERIFY_STATE_DIR: stateDir,
    },
  });
  const exitCode = res.status ?? 1;
  const record = {
    timestamp: new Date().toISOString(),
    command: smokeCmd,
    root,
    isolatedPort,
    stateDir,
    exitCode,
    error: res.error ? res.error.message : null,
    stdout: res.stdout ?? "",
    stderr: res.stderr ?? "",
  };
  if (subcmd === "evidence") {
    const outFile = join(stateDir, "latest-evidence.json");
    writeFileSync(outFile, JSON.stringify(record, null, 2) + "\n", "utf8");
    console.log("Saved runtime evidence:", outFile);
  } else {
    process.stdout.write(record.stdout);
    process.stderr.write(record.stderr);
    if (record.error) console.error(record.error);
  }
  if (res.error || exitCode !== 0) process.exit(exitCode || 1);
} else if (subcmd === "cleanup") {
  if (existsSync(stateDir)) rmSync(stateDir, { recursive: true, force: true });
  console.log("Cleaned isolated state:", stateDir);
} else {
  console.error("Unknown subcommand. Expected: doctor | drive | evidence | cleanup");
  process.exit(1);
}
