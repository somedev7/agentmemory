import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";
import { logger } from "../logger.js";
import { CliQuota, HEADLESS_RULES, parseExtraArgs } from "./agent-cli.js";
import type { QuotaLedger } from "./agent-cli.js";

// Fork-only provider (not for upstream): routes LLM work through the
// locally installed Gemini agent CLI — Antigravity (`agy`) by default,
// classic `gemini` works too — so the cost lands on the user's existing
// Google subscription instead of a per-token API. Both CLIs accept the
// payload on stdin and an instruction via `-p`, and print the bare
// response to stdout in headless mode.
//
// Serialization and the per-day quota ledger (gemini-cli-quota.json,
// daily call cap) live in the shared agent-cli machinery — see
// providers/agent-cli.ts.

const quota = new CliQuota({
  label: "gemini-cli",
  fileEnvVar: "AGENTMEMORY_GEMINI_CLI_QUOTA_FILE",
  defaultFileName: "gemini-cli-quota.json",
  capEnvVar: "AGENTMEMORY_GEMINI_CLI_DAILY_CAP",
});

export type { QuotaLedger };

export function quotaLedgerPath(): string {
  return quota.path();
}

export function readQuotaLedger(): QuotaLedger {
  return quota.read();
}

export function dailyCap(): number {
  return quota.cap();
}

const QUOTA_ERROR_RE =
  /429|RESOURCE_EXHAUSTED|quota.{0,40}(exceeded|exhausted|limit)|rate.?limit/i;

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
    const inChars = systemPrompt.length + userPrompt.length;
    quota.begin(inChars);

    const startMs = Date.now();
    try {
      const out = await this.spawnCli(systemPrompt + HEADLESS_RULES, userPrompt);
      const day = quota.success(out.length);
      logger.info("gemini-cli call ok", {
        bin: this.bin,
        latencyMs: Date.now() - startMs,
        inChars,
        outChars: out.length,
        callsToday: day.calls,
        cap: quota.cap(),
      });
      return out;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      quota.failure(msg, QUOTA_ERROR_RE);
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
