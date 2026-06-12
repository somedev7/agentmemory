import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, readFileSync, rmSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GeminiCliProvider, readQuotaLedger } from "../src/providers/gemini-cli.js";

// The provider shells out to a Gemini agent CLI (agy/gemini): payload on
// stdin, instruction via -p, bare response on stdout. A fake shell-script
// bin stands in for the real CLI so the contract (serialization, quota
// ledger, cap, error classification) is testable offline.

let dir: string;
let ledgerPath: string;

function makeFakeBin(script: string): string {
  const bin = join(dir, "fake-agy");
  writeFileSync(bin, `#!/bin/sh\n${script}\n`);
  chmodSync(bin, 0o755);
  return bin;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "gemini-cli-test-"));
  ledgerPath = join(dir, "quota.json");
  process.env.AGENTMEMORY_GEMINI_CLI_QUOTA_FILE = ledgerPath;
  delete process.env.AGENTMEMORY_GEMINI_CLI_DAILY_CAP;
});

afterEach(() => {
  delete process.env.AGENTMEMORY_GEMINI_CLI_QUOTA_FILE;
  delete process.env.AGENTMEMORY_GEMINI_CLI_DAILY_CAP;
  rmSync(dir, { recursive: true, force: true });
});

describe("GeminiCliProvider", () => {
  it("passes payload on stdin, instruction via -p, returns trimmed stdout", async () => {
    const bin = makeFakeBin(
      `payload=$(cat)\nprintf 'GOT:%s|ARG1:%s\\n' "$payload" "$1"`,
    );
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    const out = await p.compress("SYSTEM-PROMPT", "USER-PAYLOAD");

    expect(out).toContain("GOT:USER-PAYLOAD");
    expect(out).toContain("-p"); // first argv entry is the -p flag itself
  });

  it("appends headless no-tools rules to the instruction", async () => {
    const bin = makeFakeBin(`cat > /dev/null\nprintf '%s' "$2"`);
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    const out = await p.summarize("MY-SYSTEM", "x");

    expect(out).toContain("MY-SYSTEM");
    expect(out).toContain("Do not use any tools");
  });

  it("serializes concurrent calls — one CLI process at a time", async () => {
    const log = join(dir, "calls.log");
    const bin = makeFakeBin(
      `cat > /dev/null\necho "start $$" >> ${log}\nsleep 0.3\necho "end $$" >> ${log}\necho ok`,
    );
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    await Promise.all([p.compress("s", "a"), p.compress("s", "b"), p.summarize("s", "c")]);

    const lines = readFileSync(log, "utf-8").trim().split("\n");
    expect(lines).toHaveLength(6);
    for (let i = 0; i < lines.length; i += 2) {
      expect(lines[i]).toMatch(/^start /);
      expect(lines[i + 1]).toMatch(/^end /); // no interleaved start/start
    }
  });

  it("records usage in the quota ledger", async () => {
    const bin = makeFakeBin(`cat > /dev/null\necho enriched-output`);
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    await p.compress("sys", "user");
    await p.compress("sys", "user");

    const ledger = readQuotaLedger();
    const day = Object.values(ledger.days)[0];
    expect(day.calls).toBe(2);
    expect(day.ok).toBe(2);
    expect(day.failed).toBe(0);
    expect(day.outChars).toBeGreaterThan(0);
    expect(ledger.lastCallAt).toBeTruthy();
  });

  it("enforces the daily cap", async () => {
    process.env.AGENTMEMORY_GEMINI_CLI_DAILY_CAP = "1";
    const bin = makeFakeBin(`cat > /dev/null\necho ok`);
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    await p.compress("s", "u");
    await expect(p.compress("s", "u")).rejects.toThrow(/daily_cap_reached/);

    const day = Object.values(readQuotaLedger().days)[0];
    expect(day.calls).toBe(1); // capped call is refused before spawning
  });

  it("classifies quota errors from the CLI as quotaHits", async () => {
    const bin = makeFakeBin(
      `cat > /dev/null\necho "Error: 429 RESOURCE_EXHAUSTED" >&2\nexit 1`,
    );
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    await expect(p.compress("s", "u")).rejects.toThrow(/429/);

    const ledger = readQuotaLedger();
    const day = Object.values(ledger.days)[0];
    expect(day.failed).toBe(1);
    expect(day.quotaHits).toBe(1);
    expect(ledger.lastError).toContain("429");
  });

  it("rejects on empty output and on timeout", async () => {
    const empty = makeFakeBin(`cat > /dev/null\nexit 0`);
    const p1 = new GeminiCliProvider({ bin: empty, extraArgs: [] });
    await expect(p1.compress("s", "u")).rejects.toThrow(/empty output/);

    const slow = makeFakeBin(`cat > /dev/null\nsleep 5\necho late`);
    const p2 = new GeminiCliProvider({ bin: slow, extraArgs: [], timeoutMs: 300 });
    await expect(p2.compress("s", "u")).rejects.toThrow(/timed out/);
  });

  it("sets the SDK-child recursion marker for the spawned process", async () => {
    const bin = makeFakeBin(`cat > /dev/null\nprintf 'marker=%s' "$AGENTMEMORY_SDK_CHILD"`);
    const p = new GeminiCliProvider({ bin, extraArgs: [] });

    const out = await p.compress("s", "u");

    expect(out).toBe("marker=1");
  });
});
