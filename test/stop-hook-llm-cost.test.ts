import { describe, it, expect } from "vitest";
import { createServer } from "node:http";
import { resolve } from "node:path";
import { spawn } from "node:child_process";

// Stop fires after EVERY assistant turn. Until this fix it triggered two
// full-session LLM summarize passes per turn (POST /agentmemory/summarize
// plus /agentmemory/session/end → event::session::stopped → mem::summarize),
// the top driver of provider spend. Default is now a no-op; the old
// behaviour is opt-in via AGENTMEMORY_SUMMARIZE_ON_STOP=true.
const STOP_SCRIPT = resolve(__dirname, "..", "plugin", "scripts", "stop.mjs");

type ObservedRequest = { path: string; body: Record<string, unknown> };

async function runStopHook(
  payload: Record<string, unknown>,
  env: Record<string, string> = {},
): Promise<ObservedRequest[]> {
  const requests: ObservedRequest[] = [];
  const server = createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      requests.push({
        path: req.url ?? "",
        body: raw ? (JSON.parse(raw) as Record<string, unknown>) : {},
      });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ success: true }));
    });
  });

  await new Promise<void>((resolveServer) => {
    server.listen(0, "127.0.0.1", resolveServer);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("test server did not bind to a TCP port");
  }

  try {
    const childEnv: NodeJS.ProcessEnv = {
      ...process.env,
      AGENTMEMORY_URL: `http://127.0.0.1:${address.port}`,
      AGENTMEMORY_SECRET: "",
    };
    // The host running these tests may have the opt-in exported; the
    // default-behaviour case needs it absent unless the test sets it.
    delete childEnv.AGENTMEMORY_SUMMARIZE_ON_STOP;
    Object.assign(childEnv, env);

    const child = spawn(process.execPath, [STOP_SCRIPT], {
      env: childEnv,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.stdin.end(JSON.stringify(payload));

    const exitCode = await new Promise<number | null>((resolveExit, reject) => {
      const timeout = setTimeout(() => {
        child.kill();
        reject(new Error("stop hook timed out"));
      }, 5000);
      child.on("error", reject);
      child.on("close", (code) => {
        clearTimeout(timeout);
        resolveExit(code);
      });
    });
    expect(exitCode, stderr).toBe(0);
    return requests;
  } finally {
    await new Promise<void>((resolveClose) => {
      server.close(() => resolveClose());
    });
  }
}

describe("Stop hook makes no LLM-triggering requests by default", () => {
  it("default: exits without hitting /summarize or /session/end", async () => {
    const requests = await runStopHook({
      session_id: "ses_stop_default",
      entrypoint: "cli",
    });
    expect(requests).toHaveLength(0);
  });

  it("AGENTMEMORY_SUMMARIZE_ON_STOP=true restores the per-turn summarize", async () => {
    const requests = await runStopHook(
      { session_id: "ses_stop_optin", entrypoint: "cli" },
      { AGENTMEMORY_SUMMARIZE_ON_STOP: "true" },
    );
    const paths = requests.map((r) => r.path).sort();
    expect(paths).toEqual([
      "/agentmemory/session/end",
      "/agentmemory/summarize",
    ]);
    for (const r of requests) {
      expect(r.body).toMatchObject({ sessionId: "ses_stop_optin" });
    }
  });

  it("opt-in still respects the SDK-child recursion guard", async () => {
    const requests = await runStopHook(
      { session_id: "ses_stop_sdk", entrypoint: "sdk-ts" },
      { AGENTMEMORY_SUMMARIZE_ON_STOP: "true" },
    );
    expect(requests).toHaveLength(0);
  });
});
