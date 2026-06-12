import { spawn } from "node:child_process";
import { tmpdir, homedir } from "node:os";
import { dirname, join } from "node:path";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";
import { logger } from "../logger.js";

// Fork-only provider (not for upstream): routes LLM work through the
// locally installed Gemini agent CLI — Antigravity (`agy`) by default,
// classic `gemini` works too — so the cost lands on the user's existing
// Google subscription instead of a per-token API. Both CLIs accept the
// payload on stdin and an instruction via `-p`, and print the bare
// response to stdout in headless mode.
//
// Calls are strictly serialized: subscription quotas are concurrency-
// and volume-limited, and the intended workload (mem::batch-enrich every
// ~20 minutes, one summarize per session, periodic consolidation) never
// needs parallelism. A local per-day ledger persisted to
// gemini-cli-quota.json tracks usage and enforces a daily call cap,
// since the subscription side exposes no quota API.

interface QuotaDay {
  calls: number;
  ok: number;
  failed: number;
  quotaHits: number;
  inChars: number;
  outChars: number;
}

export interface QuotaLedger {
  days: Record<string, QuotaDay>;
  lastCallAt?: string;
  lastError?: string;
  updatedAt?: string;
}

const LEDGER_KEEP_DAYS = 30;

export function quotaLedgerPath(): string {
  return (
    getEnvVar("AGENTMEMORY_GEMINI_CLI_QUOTA_FILE") ||
    join(homedir(), ".agentmemory", "gemini-cli-quota.json")
  );
}

export function readQuotaLedger(): QuotaLedger {
  try {
    const raw = readFileSync(quotaLedgerPath(), "utf-8");
    const parsed = JSON.parse(raw) as QuotaLedger;
    if (parsed && typeof parsed === "object" && parsed.days) return parsed;
  } catch {
    // missing or corrupt ledger — start fresh
  }
  return { days: {} };
}

function writeQuotaLedger(ledger: QuotaLedger): void {
  try {
    const path = quotaLedgerPath();
    mkdirSync(dirname(path), { recursive: true });
    const dayKeys = Object.keys(ledger.days).sort();
    for (const key of dayKeys.slice(0, Math.max(0, dayKeys.length - LEDGER_KEEP_DAYS))) {
      delete ledger.days[key];
    }
    ledger.updatedAt = new Date().toISOString();
    writeFileSync(path, JSON.stringify(ledger, null, 2));
  } catch (err) {
    logger.warn("gemini-cli quota ledger write failed", {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

/** Local-date key — the cap is a local safety budget, not Google's window. */
function todayKey(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function emptyDay(): QuotaDay {
  return { calls: 0, ok: 0, failed: 0, quotaHits: 0, inChars: 0, outChars: 0 };
}

export function dailyCap(): number {
  const raw = getEnvVar("AGENTMEMORY_GEMINI_CLI_DAILY_CAP");
  const n = raw ? parseInt(raw, 10) : NaN;
  return Number.isFinite(n) && n > 0 ? n : 250;
}

const QUOTA_ERROR_RE =
  /429|RESOURCE_EXHAUSTED|quota.{0,40}(exceeded|exhausted|limit)|rate.?limit/i;

const HEADLESS_RULES =
  "\n\nIMPORTANT: You are running headless inside an automated pipeline. " +
  "Do not use any tools. Do not ask questions. Reply with ONLY the " +
  "requested output format — no preamble, no commentary, no markdown fences.";

export class GeminiCliProvider implements MemoryProvider {
  name = "gemini-cli";

  private bin: string;
  private extraArgs: string[];
  private timeoutMs: number;
  // Serialization chain — at most one CLI process in flight.
  private queue: Promise<unknown> = Promise.resolve();

  constructor(opts?: { bin?: string; extraArgs?: string[]; timeoutMs?: number }) {
    this.bin = opts?.bin || getEnvVar("AGENTMEMORY_GEMINI_CLI_BIN") || "agy";
    const extra = opts?.extraArgs ?? parseExtraArgs(getEnvVar("AGENTMEMORY_GEMINI_CLI_EXTRA_ARGS"));
    this.extraArgs = extra;
    const rawTimeout = getEnvVar("AGENTMEMORY_GEMINI_CLI_TIMEOUT_MS");
    const parsed = rawTimeout ? parseInt(rawTimeout, 10) : NaN;
    this.timeoutMs = opts?.timeoutMs ?? (Number.isFinite(parsed) && parsed > 0 ? parsed : 180000);
  }

  compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.enqueue(systemPrompt, userPrompt);
  }

  summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.enqueue(systemPrompt, userPrompt);
  }

  private enqueue(systemPrompt: string, userPrompt: string): Promise<string> {
    const task = () => this.invoke(systemPrompt, userPrompt);
    const next = this.queue.then(task, task);
    this.queue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async invoke(systemPrompt: string, userPrompt: string): Promise<string> {
    const ledger = readQuotaLedger();
    const key = todayKey();
    const day = (ledger.days[key] ??= emptyDay());
    const cap = dailyCap();
    if (day.calls >= cap) {
      throw new Error(
        `gemini_cli_daily_cap_reached: ${day.calls}/${cap} calls today — ` +
          `raise AGENTMEMORY_GEMINI_CLI_DAILY_CAP or wait for the next local day`,
      );
    }
    day.calls += 1;
    day.inChars += systemPrompt.length + userPrompt.length;
    ledger.lastCallAt = new Date().toISOString();
    writeQuotaLedger(ledger);

    const startMs = Date.now();
    try {
      const out = await this.spawnCli(systemPrompt + HEADLESS_RULES, userPrompt);
      const done = readQuotaLedger();
      const doneDay = (done.days[key] ??= emptyDay());
      doneDay.ok += 1;
      doneDay.outChars += out.length;
      writeQuotaLedger(done);
      logger.info("gemini-cli call ok", {
        bin: this.bin,
        latencyMs: Date.now() - startMs,
        inChars: systemPrompt.length + userPrompt.length,
        outChars: out.length,
        callsToday: doneDay.calls,
        cap,
      });
      return out;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const done = readQuotaLedger();
      const doneDay = (done.days[key] ??= emptyDay());
      doneDay.failed += 1;
      if (QUOTA_ERROR_RE.test(msg)) doneDay.quotaHits += 1;
      done.lastError = msg.slice(0, 500);
      writeQuotaLedger(done);
      throw err;
    }
  }

  private spawnCli(instruction: string, stdinPayload: string): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      // Classic gemini CLI indexes its cwd as a workspace — point it at an
      // empty directory, not tmpdir() full of unreadable system entries.
      const cwd = getEnvVar("AGENTMEMORY_GEMINI_CLI_CWD") || tmpdir();
      const child = spawn(this.bin, ["-p", instruction, ...this.extraArgs], {
        cwd,
        env: {
          ...process.env,
          // Cross-process recursion guard: any agentmemory hook inside the
          // spawned CLI sees this marker and skips its REST callbacks.
          AGENTMEMORY_SDK_CHILD: "1",
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      let settled = false;

      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.kill("SIGKILL");
        reject(
          new Error(`gemini-cli timed out after ${this.timeoutMs}ms (bin: ${this.bin})`),
        );
      }, this.timeoutMs);

      child.stdout.on("data", (chunk) => {
        stdout += chunk;
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk;
      });
      child.on("error", (err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(new Error(`gemini-cli spawn failed (bin: ${this.bin}): ${err.message}`));
      });
      child.on("close", (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const trimmed = stdout.trim();
        if (code !== 0) {
          reject(
            new Error(
              `gemini-cli exited with code ${code}: ${(stderr || trimmed).slice(0, 500)}`,
            ),
          );
          return;
        }
        if (!trimmed) {
          reject(new Error(`gemini-cli returned empty output: ${stderr.slice(0, 300)}`));
          return;
        }
        resolve(trimmed);
      });

      child.stdin.on("error", () => {
        // CLI may exit before consuming stdin; close handler reports it.
      });
      child.stdin.end(stdinPayload);
    });
  }
}

function parseExtraArgs(raw: string | undefined): string[] {
  if (!raw || !raw.trim()) return [];
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
