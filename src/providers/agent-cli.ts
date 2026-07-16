import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { getEnvVar } from "../config.js";
import { logger } from "../logger.js";

// Fork-only shared machinery for subscription-billed agent-CLI providers
// (currently codex-cli). Each shells out to a locally installed agent CLI
// and bill the user's existing subscription, which exposes no usage API —
// so each provider keeps a local per-day ledger and refuses calls past a
// daily cap. Calls are strictly serialized by the providers themselves:
// subscription quotas are concurrency- and volume-limited, and the
// intended workload (mem::batch-enrich every ~20 minutes, one summarize
// per session, periodic consolidation) never needs parallelism.

export interface QuotaDay {
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
const DEFAULT_DAILY_CAP = 250;

/** Local-date key — the cap is a local safety budget, not the vendor's window. */
export function todayKey(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function emptyDay(): QuotaDay {
  return { calls: 0, ok: 0, failed: 0, quotaHits: 0, inChars: 0, outChars: 0 };
}

export interface CliQuotaOptions {
  /** Provider name used in log lines and the cap error prefix, e.g. "codex-cli". */
  label: string;
  fileEnvVar: string;
  defaultFileName: string;
  capEnvVar: string;
}

export class CliQuota {
  constructor(private opts: CliQuotaOptions) {}

  path(): string {
    return (
      getEnvVar(this.opts.fileEnvVar) ||
      join(homedir(), ".agentmemory", this.opts.defaultFileName)
    );
  }

  read(): QuotaLedger {
    try {
      const raw = readFileSync(this.path(), "utf-8");
      const parsed = JSON.parse(raw) as QuotaLedger;
      if (parsed && typeof parsed === "object" && parsed.days) return parsed;
    } catch {
      // missing or corrupt ledger — start fresh
    }
    return { days: {} };
  }

  write(ledger: QuotaLedger): void {
    try {
      const path = this.path();
      mkdirSync(dirname(path), { recursive: true });
      const dayKeys = Object.keys(ledger.days).sort();
      for (const key of dayKeys.slice(0, Math.max(0, dayKeys.length - LEDGER_KEEP_DAYS))) {
        delete ledger.days[key];
      }
      ledger.updatedAt = new Date().toISOString();
      writeFileSync(path, JSON.stringify(ledger, null, 2));
    } catch (err) {
      logger.warn(`${this.opts.label} quota ledger write failed`, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  cap(): number {
    const raw = getEnvVar(this.opts.capEnvVar);
    const n = raw ? parseInt(raw, 10) : NaN;
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_DAILY_CAP;
  }

  /** Cap check + call/inChars accounting. Throws before any spawn past the cap. */
  begin(inChars: number): void {
    const ledger = this.read();
    const day = (ledger.days[todayKey()] ??= emptyDay());
    const cap = this.cap();
    if (day.calls >= cap) {
      throw new Error(
        `${this.opts.label.replace(/-/g, "_")}_daily_cap_reached: ${day.calls}/${cap} calls today — ` +
          `raise ${this.opts.capEnvVar} or wait for the next local day`,
      );
    }
    day.calls += 1;
    day.inChars += inChars;
    ledger.lastCallAt = new Date().toISOString();
    this.write(ledger);
  }

  /** Re-reads the ledger (other writes may have landed) and records success. */
  success(outChars: number): QuotaDay {
    const ledger = this.read();
    const day = (ledger.days[todayKey()] ??= emptyDay());
    day.ok += 1;
    day.outChars += outChars;
    this.write(ledger);
    return day;
  }

  failure(message: string, quotaErrorRe: RegExp): void {
    const ledger = this.read();
    const day = (ledger.days[todayKey()] ??= emptyDay());
    day.failed += 1;
    if (quotaErrorRe.test(message)) day.quotaHits += 1;
    ledger.lastError = message.slice(0, 500);
    this.write(ledger);
  }
}

export const HEADLESS_RULES =
  "\n\nIMPORTANT: You are running headless inside an automated pipeline. " +
  "Do not use any tools. Do not ask questions. Reply with ONLY the " +
  "requested output format — no preamble, no commentary, no markdown fences.";

export function parseExtraArgs(raw: string | undefined): string[] {
  if (!raw || !raw.trim()) return [];
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
