import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("../src/state/schema.js", () => ({
  KV: {
    sessions: "sessions",
    observations: (sessionId: string) => `obs:${sessionId}`,
    audit: "audit",
  },
}));

vi.mock("../src/functions/audit.js", () => ({
  safeAudit: vi.fn(),
}));

const searchAdds: unknown[] = [];
const vectorAdds: Array<{ id: string }> = [];
vi.mock("../src/functions/search.js", () => ({
  getSearchIndex: () => ({
    add: (obs: unknown) => {
      searchAdds.push(obs);
    },
  }),
  vectorIndexAddGuarded: async (id: string) => {
    vectorAdds.push({ id });
    return true;
  },
}));

import {
  registerBatchEnrichFunction,
  isSyntheticCandidate,
  parseEnrichmentResponse,
} from "../src/functions/batch-enrich.js";
import type {
  CompressedObservation,
  MemoryProvider,
  Session,
} from "../src/types.js";

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
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

function mockSdk() {
  const functions = new Map<string, Function>();
  return {
    functions,
    registerFunction: (id: string, handler: Function) => {
      functions.set(id, handler);
    },
    registerTrigger: () => {},
    trigger: async () => ({}),
  };
}

function syntheticObs(i: number, sessionId: string): CompressedObservation {
  return {
    id: `obs_${i}`,
    sessionId,
    timestamp: new Date(Date.now() - i * 1000).toISOString(),
    type: "command_run",
    title: "Bash",
    subtitle: "git status",
    facts: [],
    narrative: "git status | clean tree",
    concepts: [],
    files: [],
    importance: 5,
    confidence: 0.3,
    synthetic: true,
  };
}

function enrichmentXml(ids: string[]): string {
  return ids
    .map(
      (id) => `<obs id="${id}">
<title>Checked git status</title>
<facts><fact>Working tree was clean</fact></facts>
<narrative>Ran git status; the tree was clean.</narrative>
<concepts><concept>git</concept></concepts>
<importance>4</importance>
</obs>`,
    )
    .join("\n");
}

function makeProvider(respond: (user: string) => string): MemoryProvider & {
  calls: string[];
} {
  const calls: string[] = [];
  return {
    name: "gemini-cli",
    calls,
    compress: async (_system: string, user: string) => {
      calls.push(user);
      return respond(user);
    },
    summarize: async () => "",
  };
}

async function setup(opts: {
  obs: CompressedObservation[];
  provider: MemoryProvider;
  sessionUpdatedAt?: string;
}) {
  const sdk = mockSdk();
  const kv = mockKV();
  const session: Session = {
    id: "ses1",
    project: "p",
    cwd: "/tmp",
    startedAt: new Date().toISOString(),
    status: "active",
    observationCount: opts.obs.length,
  };
  (session as { updatedAt?: string }).updatedAt =
    opts.sessionUpdatedAt ?? new Date().toISOString();
  await kv.set("sessions", "ses1", session);
  for (const o of opts.obs) {
    await kv.set("obs:ses1", o.id, o);
  }
  registerBatchEnrichFunction(sdk as any, kv as any, opts.provider);
  return { handler: sdk.functions.get("mem::batch-enrich")!, kv };
}

beforeEach(() => {
  searchAdds.length = 0;
  vectorAdds.length = 0;
  delete process.env.AGENTMEMORY_BATCH_ENRICH_MAX;
});

describe("isSyntheticCandidate", () => {
  it("matches flagged synthetic records and the pre-flag heuristic shape", () => {
    expect(isSyntheticCandidate(syntheticObs(1, "s"))).toBe(true);
    const legacy = { ...syntheticObs(2, "s") };
    delete (legacy as { synthetic?: boolean }).synthetic;
    expect(isSyntheticCandidate(legacy)).toBe(true);
  });

  it("skips enriched, LLM-compressed and image records", () => {
    expect(
      isSyntheticCandidate({ ...syntheticObs(1, "s"), enrichedAt: "2026-01-01" }),
    ).toBe(false);
    expect(
      isSyntheticCandidate({
        ...syntheticObs(2, "s"),
        synthetic: undefined,
        confidence: 0.9,
        facts: ["a"],
      }),
    ).toBe(false);
    expect(
      isSyntheticCandidate({ ...syntheticObs(3, "s"), modality: "image" }),
    ).toBe(false);
  });
});

describe("mem::batch-enrich", () => {
  it("enriches synthetic observations in one provider call and reindexes", async () => {
    const provider = makeProvider(() => enrichmentXml(["obs_1", "obs_2"]));
    const { handler, kv } = await setup({
      obs: [syntheticObs(1, "ses1"), syntheticObs(2, "ses1")],
      provider,
    });

    const result: any = await handler({});

    expect(result.success).toBe(true);
    expect(result.enriched).toBe(2);
    expect(provider.calls).toHaveLength(1); // ONE call for the whole batch
    const stored = (await kv.get("obs:ses1", "obs_1")) as CompressedObservation;
    expect(stored.title).toBe("Checked git status");
    expect(stored.facts).toEqual(["Working tree was clean"]);
    expect(stored.synthetic).toBe(false);
    expect(stored.enrichedAt).toBeTruthy();
    expect(stored.confidence).toBe(0.7);
    expect(searchAdds).toHaveLength(2);
    expect(vectorAdds.map((v) => v.id).sort()).toEqual(["obs_1", "obs_2"]);
  });

  it("leaves unparsed observations synthetic for a later tick", async () => {
    const provider = makeProvider(() => enrichmentXml(["obs_1"])); // obs_2 missing
    const { handler, kv } = await setup({
      obs: [syntheticObs(1, "ses1"), syntheticObs(2, "ses1")],
      provider,
    });

    const result: any = await handler({});

    expect(result.enriched).toBe(1);
    expect(result.unparsed).toBe(1);
    const left = (await kv.get("obs:ses1", "obs_2")) as CompressedObservation;
    expect(left.synthetic).toBe(true);
    expect(left.enrichedAt).toBeUndefined();
  });

  it("respects the batch limit and reports the remaining backlog", async () => {
    const provider = makeProvider((user) => {
      const ids = [...user.matchAll(/<input id="([^"]+)"/g)].map((m) => m[1]);
      return enrichmentXml(ids);
    });
    const obs = Array.from({ length: 5 }, (_, i) => syntheticObs(i, "ses1"));
    const { handler } = await setup({ obs, provider });

    const result: any = await handler({ limit: 2 });

    expect(result.requested).toBe(2);
    expect(result.enriched).toBe(2);
    expect(result.pending).toBe(3);
  });

  it("returns the provider error without touching records", async () => {
    const provider: MemoryProvider = {
      name: "gemini-cli",
      compress: async () => {
        throw new Error("gemini_cli_daily_cap_reached: 250/250");
      },
      summarize: async () => "",
    };
    const { handler, kv } = await setup({
      obs: [syntheticObs(1, "ses1")],
      provider,
    });

    const result: any = await handler({});

    expect(result.success).toBe(false);
    expect(result.error).toContain("daily_cap_reached");
    const stored = (await kv.get("obs:ses1", "obs_1")) as CompressedObservation;
    expect(stored.synthetic).toBe(true);
  });

  it("ignores sessions outside the lookback window", async () => {
    const provider = makeProvider(() => enrichmentXml(["obs_1"]));
    const old = new Date(Date.now() - 100 * 3600_000).toISOString();
    const { handler } = await setup({
      obs: [syntheticObs(1, "ses1")],
      provider,
      sessionUpdatedAt: old,
    });

    const result: any = await handler({});

    expect(result.requested).toBe(0);
    expect((provider as { calls: string[] }).calls).toHaveLength(0);
  });
});

describe("parseEnrichmentResponse", () => {
  it("parses multiple obs blocks and skips ones without a title", () => {
    const xml =
      enrichmentXml(["a"]) + '\n<obs id="b"><narrative>no title</narrative></obs>';
    const parsed = parseEnrichmentResponse(xml);
    expect(parsed.has("a")).toBe(true);
    expect(parsed.has("b")).toBe(false);
    expect(parsed.get("a")!.importance).toBe(4);
  });
});
