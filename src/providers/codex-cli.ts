import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readFileSync, rmSync } from "node:fs";
import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";
import { logger } from "../logger.js";
import { CliQuota, HEADLESS_RULES, parseExtraArgs } from "./agent-cli.js";
import type { QuotaLedger } from "./agent-cli.js";

// Fork-only provider (not for upstream): routes LLM work through the
// locally installed OpenAI Codex CLI (`codex exec`) so the cost lands on
// the user's ChatGPT subscription instead of a per-token API. Replaced
// the earlier gemini-cli provider (removed 2026-07-16): Google retired
// the consumer Gemini CLI and Antigravity weekly quotas are too small
// for the background workload.
//
// Invocation shape: the instruction goes as
// the positional argument, the payload is piped to stdin (codex appends
// it as a `<stdin>` block), and the final agent message is read from a
// temp file via -o — stdout carries the noisy event log, not the answer.
// --ignore-user-config keeps the user's interactive codex setup (frontier
// model, xhigh reasoning, plugins, MCP servers) out of pipeline calls;
// auth is still read from CODEX_HOME. The model is pinned to the cheapest
// listed slug at low reasoning effort: background memory work needs
// format compliance, not depth, and the quota pool is shared with the
// user's interactive codex sessions.
//
// Serialization and the per-day quota ledger (codex-cli-quota.json,
// daily call cap) live in the shared agent-cli machinery — see
// providers/agent-cli.ts.

const quota = new CliQuota({
  label: "codex-cli",
  fileEnvVar: "AGENTMEMORY_CODEX_CLI_QUOTA_FILE",
  defaultFileName: "codex-cli-quota.json",
  capEnvVar: "AGENTMEMORY_CODEX_CLI_DAILY_CAP",
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

// ChatGPT-side limit errors: HTTP 429s plus "You've hit your usage limit" /
// "usage_limit_reached" phrasings from the Codex backend.
const QUOTA_ERROR_RE = /429|usage.?limit|quota|rate.?limit|too many requests/i;

let outFileSeq = 0;

export class CodexCliProvider implements MemoryProvider {
  name = "codex-cli";

  private bin: string;
  private model: string;
  private reasoning: string;
  private extraArgs: string[];
  private timeoutMs: number;
  // Serialization chain — at most one CLI process in flight.
  private queue: Promise<unknown> = Promise.resolve();

  constructor(opts?: {
    bin?: string;
    model?: string;
    reasoning?: string;
    extraArgs?: string[];
    timeoutMs?: number;
  }) {
    this.bin = opts?.bin || getEnvVar("AGENTMEMORY_CODEX_CLI_BIN") || "codex";
    this.model = opts?.model || getEnvVar("AGENTMEMORY_CODEX_CLI_MODEL") || "gpt-5.4-mini";
    this.reasoning = opts?.reasoning || getEnvVar("AGENTMEMORY_CODEX_CLI_REASONING") || "low";
    this.extraArgs = opts?.extraArgs ?? parseExtraArgs(getEnvVar("AGENTMEMORY_CODEX_CLI_EXTRA_ARGS"));
    const rawTimeout = getEnvVar("AGENTMEMORY_CODEX_CLI_TIMEOUT_MS");
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
      logger.info("codex-cli call ok", {
        bin: this.bin,
        model: this.model,
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
      const outFile = join(tmpdir(), `agentmemory-codex-${process.pid}-${++outFileSeq}.out`);
      const cwd = getEnvVar("AGENTMEMORY_CODEX_CLI_CWD") || tmpdir();
      const args = [
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-m",
        this.model,
        "-c",
        `model_reasoning_effort="${this.reasoning}"`,
        "-o",
        outFile,
        ...this.extraArgs,
        instruction,
      ];
      const child = spawn(this.bin, args, {
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

      const cleanup = () => {
        try {
          rmSync(outFile, { force: true });
        } catch {
          // best-effort temp file removal
        }
      };

      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.kill("SIGKILL");
        cleanup();
        reject(
          new Error(`codex-cli timed out after ${this.timeoutMs}ms (bin: ${this.bin})`),
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
        cleanup();
        reject(new Error(`codex-cli spawn failed (bin: ${this.bin}): ${err.message}`));
      });
      child.on("close", (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        // Errors land at the END of the stream (codex prints a header block
        // first, then ERROR lines), so keep the tail, not the head.
        const detail = (stderr.trim() || stdout.trim()).slice(-500);
        if (code !== 0) {
          cleanup();
          reject(new Error(`codex-cli exited with code ${code}: ${detail}`));
          return;
        }
        let lastMessage = "";
        try {
          lastMessage = readFileSync(outFile, "utf-8").trim();
        } catch {
          // missing -o file is reported as empty output below
        }
        cleanup();
        if (!lastMessage) {
          reject(new Error(`codex-cli returned empty output: ${detail.slice(-300)}`));
          return;
        }
        resolve(lastMessage);
      });

      child.stdin.on("error", () => {
        // CLI may exit before consuming stdin; close handler reports it.
      });
      child.stdin.end(stdinPayload);
    });
  }
}
