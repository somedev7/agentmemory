import { describe, it, expect, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("../src/functions/slots.js", () => ({
  isReflectEnabled: () => false,
}));

vi.mock("../src/config.js", () => ({
  isGraphExtractionEnabled: () => true,
  isConsolidationEnabled: () => false,
  getConsolidationCooldownMs: () => 0,
  getAgentId: () => undefined,
}));

vi.mock("../src/state/schema.js", () => ({
  KV: {
    sessions: "sessions",
    observations: (sessionId: string) => `obs:${sessionId}`,
  },
  STREAM: { name: "stream", viewerGroup: "viewer" },
}));

import { registerEventTriggers } from "../src/triggers/events.js";
import type { CompressedObservation, Session } from "../src/types.js";

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    store,
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    update: async (
      scope: string,
      key: string,
      ops: Array<{ type: string; path: string; value: unknown }>,
    ) => {
      const current = (store.get(scope)?.get(key) ?? {}) as Record<string, unknown>;
      for (const op of ops) {
        if (op.type === "set") current[op.path] = op.value;
      }
      store.get(scope)?.set(key, current);
      return current;
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

function mockSdk() {
  const functions = new Map<string, Function>();
  const triggers: Array<{ function_id: string; payload: unknown }> = [];
  return {
    functions,
    triggers,
    registerFunction: (id: string, handler: Function) => {
      functions.set(id, handler);
    },
    registerTrigger: () => {},
    trigger: async (call: { function_id: string; payload: unknown }) => {
      triggers.push(call);
      return {};
    },
  };
}

function makeObs(i: number, sessionId: string, timestamp: string): CompressedObservation {
  return {
    id: `obs_${i}`,
    sessionId,
    timestamp,
    type: "conversation",
    title: `obs ${i}`,
    facts: [],
    narrative: "",
    concepts: [],
    files: [],
    importance: 5,
  } as CompressedObservation;
}

async function setup(obsTimestamps: string[]) {
  const sdk = mockSdk();
  const kv = mockKV();
  const session: Session = {
    id: "ses1",
    project: "p",
    cwd: "/tmp",
    startedAt: "2026-06-12T00:00:00.000Z",
    status: "active",
    observationCount: obsTimestamps.length,
  };
  await kv.set("sessions", "ses1", session);
  for (let i = 0; i < obsTimestamps.length; i++) {
    const o = makeObs(i, "ses1", obsTimestamps[i]);
    await kv.set("obs:ses1", o.id, o);
  }
  registerEventTriggers(sdk as any, kv as any);
  const stopped = sdk.functions.get("event::session::stopped")!;
  return { sdk, kv, stopped };
}

function graphCalls(sdk: ReturnType<typeof mockSdk>) {
  return sdk.triggers.filter((t) => t.function_id === "mem::graph-extract");
}

describe("event::session::stopped — graph-extract delta watermark", () => {
  it("first run sends all observations and stores the watermark", async () => {
    const { sdk, kv, stopped } = await setup([
      "2026-06-12T00:01:00.000Z",
      "2026-06-12T00:02:00.000Z",
      "2026-06-12T00:03:00.000Z",
    ]);

    await stopped({ sessionId: "ses1" });

    const calls = graphCalls(sdk);
    expect(calls).toHaveLength(1);
    expect((calls[0].payload as { observations: unknown[] }).observations).toHaveLength(3);
    const session = await kv.get<Session>("sessions", "ses1");
    expect(session?.graphExtractedAt).toBe("2026-06-12T00:03:00.000Z");
  });

  it("second run with no new observations does not re-extract", async () => {
    const { sdk, stopped } = await setup(["2026-06-12T00:01:00.000Z"]);

    await stopped({ sessionId: "ses1" });
    await stopped({ sessionId: "ses1" });

    expect(graphCalls(sdk)).toHaveLength(1);
  });

  it("only observations newer than the watermark are sent on re-run", async () => {
    const { sdk, kv, stopped } = await setup([
      "2026-06-12T00:01:00.000Z",
      "2026-06-12T00:02:00.000Z",
    ]);

    await stopped({ sessionId: "ses1" });

    const late = makeObs(99, "ses1", "2026-06-12T00:05:00.000Z");
    await kv.set("obs:ses1", late.id, late);
    await stopped({ sessionId: "ses1" });

    const calls = graphCalls(sdk);
    expect(calls).toHaveLength(2);
    const second = (calls[1].payload as { observations: CompressedObservation[] }).observations;
    expect(second).toHaveLength(1);
    expect(second[0].id).toBe("obs_99");
    const session = await kv.get<Session>("sessions", "ses1");
    expect(session?.graphExtractedAt).toBe("2026-06-12T00:05:00.000Z");
  });
});
