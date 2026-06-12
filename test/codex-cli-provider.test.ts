import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, readFileSync, rmSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CodexCliProvider, readQuotaLedger } from "../src/providers/codex-cli.js";

// The provider shells out to `codex exec`: instruction as the positional
// arg, payload on stdin, final agent message read from the -o temp file
// (stdout carries the noisy event log, not the answer). A fake shell-script
// bin stands in for the real CLI so the contract (arg shape, -o output,
// serialization, quota ledger, cap, error classification) is testable
// offline.

let dir: string;
let ledgerPath: string;

// Every fake bin starts by reading the payload from stdin and locating the
// -o output-file argument, mirroring how the real codex exec behaves.
const FAKE_PRELUDE = `payload=$(cat)
out=""
prev=""
for a in "$@"; do
  [ "$prev" = "-o" ] && out="$a"
  prev="$a"
done`;

function makeFakeBin(script: string): string {
  const bin = join(dir, "fake-codex");
  writeFileSync(bin, `#!/bin/sh\n${FAKE_PRELUDE}\n${script}\n`);
  chmodSync(bin, 0o755);
  return bin;
}

function makeProvider(bin: string, timeoutMs?: number): CodexCliProvider {
  return new CodexCliProvider({
    bin,
    model: "gpt-5.4-mini",
    reasoning: "low",
    extraArgs: [],
    ...(timeoutMs ? { timeoutMs } : {}),
  });
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "codex-cli-test-"));
  ledgerPath = join(dir, "quota.json");
  process.env.AGENTMEMORY_CODEX_CLI_QUOTA_FILE = ledgerPath;
  delete process.env.AGENTMEMORY_CODEX_CLI_DAILY_CAP;
});

afterEach(() => {
  delete process.env.AGENTMEMORY_CODEX_CLI_QUOTA_FILE;
  delete process.env.AGENTMEMORY_CODEX_CLI_DAILY_CAP;
  rmSync(dir, { recursive: true, force: true });
});

describe("CodexCliProvider", () => {
  it("returns the -o file content, not the noisy stdout event log", async () => {
    const bin = makeFakeBin(
      `echo "header noise / tokens used: 9999"\nprintf 'GOT:%s' "$payload" > "$out"`,
    );
    const p = makeProvider(bin);

    const out = await p.compress("SYSTEM-PROMPT", "USER-PAYLOAD");

    expect(out).toBe("GOT:USER-PAYLOAD");
  });

  it("invokes codex exec headless with pinned model and reasoning effort", async () => {
    // \x1f separator: the instruction arg itself contains newlines.
    const bin = makeFakeBin(`printf '%s\\037' "$@" > "$out"`);
    const p = makeProvider(bin);

    const out = await p.summarize("MY-SYSTEM", "x");
    const args = out.split("\x1f").filter(Boolean); // printf leaves a trailing separator

    expect(args[0]).toBe("exec");
    expect(args).toContain("--ignore-user-config");
    expect(args).toContain("--skip-git-repo-check");
    expect(args).toContain("--ephemeral");
    expect(args).toContain("read-only");
    expect(args).toContain("gpt-5.4-mini");
    expect(args).toContain('model_reasoning_effort="low"');
    // Instruction is the last positional arg, with headless rules appended.
    expect(args[args.length - 1]).toContain("MY-SYSTEM");
    expect(args[args.length - 1]).toContain("Do not use any tools");
  });

  it("appends configured extra args before the instruction", async () => {
    const bin = makeFakeBin(`printf '%s\\037' "$@" > "$out"`);
    const p = new CodexCliProvider({
      bin,
      model: "gpt-5.4-mini",
      reasoning: "low",
      extraArgs: ["--profile", "memory"],
    });

    const out = await p.compress("s", "u");
    const args = out.split("\x1f").filter(Boolean);

    const profileIdx = args.indexOf("--profile");
    expect(profileIdx).toBeGreaterThan(0);
    expect(args[profileIdx + 1]).toBe("memory");
    expect(profileIdx).toBeLessThan(args.length - 1);
  });

  it("serializes concurrent calls — one CLI process at a time", async () => {
    const log = join(dir, "calls.log");
    const bin = makeFakeBin(
      `echo "start $$" >> ${log}\nsleep 0.3\necho "end $$" >> ${log}\necho ok > "$out"`,
    );
    const p = makeProvider(bin);

    await Promise.all([p.compress("s", "a"), p.compress("s", "b"), p.summarize("s", "c")]);

    const lines = readFileSync(log, "utf-8").trim().split("\n");
    expect(lines).toHaveLength(6);
    for (let i = 0; i < lines.length; i += 2) {
      expect(lines[i]).toMatch(/^start /);
      expect(lines[i + 1]).toMatch(/^end /); // no interleaved start/start
    }
  });

  it("records usage in the quota ledger", async () => {
    const bin = makeFakeBin(`echo enriched-output > "$out"`);
    const p = makeProvider(bin);

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
    process.env.AGENTMEMORY_CODEX_CLI_DAILY_CAP = "1";
    const bin = makeFakeBin(`echo ok > "$out"`);
    const p = makeProvider(bin);

    await p.compress("s", "u");
    await expect(p.compress("s", "u")).rejects.toThrow(/codex_cli_daily_cap_reached/);

    const day = Object.values(readQuotaLedger().days)[0];
    expect(day.calls).toBe(1); // capped call is refused before spawning
  });

  it("classifies ChatGPT usage-limit errors as quotaHits", async () => {
    const bin = makeFakeBin(
      `echo "ERROR: You've hit your usage limit. Upgrade to Pro." >&2\nexit 1`,
    );
    const p = makeProvider(bin);

    await expect(p.compress("s", "u")).rejects.toThrow(/usage limit/);

    const ledger = readQuotaLedger();
    const day = Object.values(ledger.days)[0];
    expect(day.failed).toBe(1);
    expect(day.quotaHits).toBe(1);
    expect(ledger.lastError).toContain("usage limit");
  });

  it("keeps the tail of stdout in errors — codex prints ERROR lines last", async () => {
    const bin = makeFakeBin(
      `echo "header block"\necho 'ERROR: {"status":429,"message":"rate limit"}'\nexit 1`,
    );
    const p = makeProvider(bin);

    await expect(p.compress("s", "u")).rejects.toThrow(/429/);

    const day = Object.values(readQuotaLedger().days)[0];
    expect(day.quotaHits).toBe(1);
  });

  it("rejects on empty -o output and on timeout", async () => {
    const empty = makeFakeBin(`echo "event log noise"\nexit 0`);
    const p1 = makeProvider(empty);
    await expect(p1.compress("s", "u")).rejects.toThrow(/empty output/);

    const slow = makeFakeBin(`sleep 5\necho late > "$out"`);
    const p2 = makeProvider(slow, 300);
    await expect(p2.compress("s", "u")).rejects.toThrow(/timed out/);
  });

  it("sets the SDK-child recursion marker for the spawned process", async () => {
    const bin = makeFakeBin(`printf 'marker=%s' "$AGENTMEMORY_SDK_CHILD" > "$out"`);
    const p = makeProvider(bin);

    const out = await p.compress("s", "u");

    expect(out).toBe("marker=1");
  });
});
