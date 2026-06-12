import type { ISdk } from "iii-sdk";
import type {
  CompressedObservation,
  MemoryProvider,
  Session,
} from "../types.js";
import { KV } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import { getXmlTag, getXmlChildren } from "../prompts/xml.js";
import { getEnvVar } from "../config.js";
import { getSearchIndex, vectorIndexAddGuarded } from "./search.js";
import { safeAudit } from "./audit.js";
import { logger } from "../logger.js";

// Fork-only (no upstream PR): deferred batch enrichment. The default
// zero-LLM path stores synthetic compressions immediately — search and
// recall work, but titles/facts/concepts are heuristic. Instead of one
// LLM call per observation (AGENTMEMORY_AUTO_COMPRESS, ~thousands of
// calls/day), a periodic tick gathers the synthetic backlog and upgrades
// it in ONE provider call, sized for a subscription-billed CLI provider
// (see providers/gemini-cli.ts).

const BATCH_ENRICH_SYSTEM = `You are a memory-compression assistant for a coding-agent observation log.
Each <input> element below is one observation from an AI coding agent's session: a tool call, command, file access, or conversation snippet. The current text is a raw mechanical capture.

For EVERY <input id="...">, output one <obs id="..."> element with the SAME id:

<obs id="the-input-id">
<title>Specific, informative title, max 10 words</title>
<facts>
<fact>One concrete fact worth recalling later (paths, commands, versions, results)</fact>
</facts>
<narrative>1-2 sentences: what happened and why it matters</narrative>
<concepts>
<concept>short-keyword</concept>
</concepts>
<importance>1-10 (10 = key decision/discovery, 5 = routine but useful, 1 = noise)</importance>
</obs>

Rules:
- Output ONLY <obs> elements, one per input, nothing else.
- Keep every id exactly as given.
- 0-4 facts and 0-4 concepts per observation; omit empty items rather than padding.
- Be specific: prefer file names, commands, error texts over generic phrasing.`;

function xmlEscape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function intEnv(name: string, fallback: number, min = 1): number {
  const raw = getEnvVar(name);
  const n = raw ? parseInt(raw, 10) : NaN;
  return Number.isFinite(n) && n >= min ? n : fallback;
}

/**
 * Records produced by buildSyntheticCompression before the explicit
 * `synthetic` flag existed are recognizable by their fixed heuristic
 * shape: confidence 0.3 and no extracted facts/concepts.
 */
export function isSyntheticCandidate(o: CompressedObservation): boolean {
  if (!o.title) return false;
  if (o.modality === "image") return false;
  if (o.enrichedAt) return false;
  if (o.synthetic === true) return true;
  return (
    o.confidence === 0.3 &&
    (o.facts?.length ?? 0) === 0 &&
    (o.concepts?.length ?? 0) === 0
  );
}

function buildInputBlock(o: CompressedObservation): string {
  const parts = [o.title, o.subtitle, o.narrative].filter(
    (s): s is string => typeof s === "string" && s.length > 0,
  );
  const files = o.files?.length ? ` files="${xmlEscape(o.files.join(","))}"` : "";
  return `<input id="${o.id}" type="${o.type}"${files}>${xmlEscape(parts.join(" | "))}</input>`;
}

interface ParsedEnrichment {
  title: string;
  facts: string[];
  narrative: string;
  concepts: string[];
  importance?: number;
}

export function parseEnrichmentResponse(
  response: string,
): Map<string, ParsedEnrichment> {
  const out = new Map<string, ParsedEnrichment>();
  const blockRe = /<obs\s+id="([^"]+)"\s*>([\s\S]*?)<\/obs>/g;
  let m: RegExpExecArray | null;
  while ((m = blockRe.exec(response)) !== null) {
    const id = m[1];
    const body = m[2];
    const title = getXmlTag(body, "title");
    if (!title || !title.trim()) continue;
    const importanceRaw = getXmlTag(body, "importance");
    const importanceParsed = importanceRaw ? parseInt(importanceRaw, 10) : NaN;
    out.set(id, {
      title: title.trim().slice(0, 120),
      facts: getXmlChildren(body, "facts", "fact").slice(0, 6),
      narrative: (getXmlTag(body, "narrative") || "").trim().slice(0, 600),
      concepts: getXmlChildren(body, "concepts", "concept").slice(0, 6),
      importance:
        Number.isFinite(importanceParsed) && importanceParsed >= 1 && importanceParsed <= 10
          ? importanceParsed
          : undefined,
    });
  }
  return out;
}

export function registerBatchEnrichFunction(
  sdk: ISdk,
  kv: StateKV,
  provider: MemoryProvider,
): void {
  sdk.registerFunction(
    "mem::batch-enrich",
    async (data?: { limit?: number; sessionId?: string }) => {
      // The daemon wraps providers as "resilient(<name>)" — match both.
      if (provider.name === "noop" || provider.name.includes("(noop)")) {
        return { success: false, error: "no_provider" };
      }

      const batchMax = data?.limit ?? intEnv("AGENTMEMORY_BATCH_ENRICH_MAX", 150);
      const lookbackH = intEnv("AGENTMEMORY_BATCH_ENRICH_LOOKBACK_H", 72);
      const cutoff = new Date(Date.now() - lookbackH * 3600_000).toISOString();

      // Gather the synthetic backlog from recently active sessions.
      let sessionIds: string[];
      if (data?.sessionId) {
        sessionIds = [data.sessionId];
      } else {
        const sessions = await kv.list<Session>(KV.sessions);
        sessionIds = sessions
          .filter((s) => {
            const last =
              (s as { updatedAt?: string }).updatedAt ?? s.endedAt ?? s.startedAt;
            return typeof last === "string" && last >= cutoff;
          })
          .map((s) => s.id);
      }

      const candidates: CompressedObservation[] = [];
      for (const sid of sessionIds) {
        const observations = await kv.list<CompressedObservation>(
          KV.observations(sid),
        );
        for (const o of observations) {
          if (isSyntheticCandidate(o)) candidates.push(o);
        }
      }
      candidates.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

      const batch = candidates.slice(0, batchMax);
      const pending = candidates.length - batch.length;
      if (batch.length === 0) {
        return { success: true, enriched: 0, requested: 0, pending: 0 };
      }

      const userPrompt = batch.map(buildInputBlock).join("\n");
      const startMs = Date.now();

      let response: string;
      try {
        response = await provider.compress(BATCH_ENRICH_SYSTEM, userPrompt);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn("batch-enrich provider call failed", {
          batch: batch.length,
          error: msg,
        });
        return { success: false, error: msg, requested: batch.length, pending };
      }

      const parsed = parseEnrichmentResponse(response || "");
      let enriched = 0;
      const enrichedIds: string[] = [];
      const now = new Date().toISOString();

      for (const o of batch) {
        const e = parsed.get(o.id);
        if (!e) continue; // stays synthetic; retried on a later tick
        const updated: CompressedObservation = {
          ...o,
          title: e.title,
          facts: e.facts,
          narrative: e.narrative || o.narrative,
          concepts: e.concepts,
          importance: e.importance ?? o.importance,
          confidence: 0.7,
          synthetic: false,
          enrichedAt: now,
        };
        await kv.set(KV.observations(o.sessionId), o.id, updated);
        // Both indices are id-keyed; re-adding replaces the synthetic entry.
        getSearchIndex().add(updated);
        await vectorIndexAddGuarded(
          updated.id,
          updated.sessionId,
          updated.title + " " + (updated.narrative || ""),
          { kind: "observation", logId: updated.id },
        );
        enriched += 1;
        enrichedIds.push(o.id);
      }

      if (enrichedIds.length > 0) {
        await safeAudit(kv, "compress", "mem::batch-enrich", enrichedIds, {
          enriched,
          requested: batch.length,
          provider: provider.name,
        });
      }

      const latencyMs = Date.now() - startMs;
      logger.info("batch-enrich tick done", {
        enriched,
        requested: batch.length,
        unparsed: batch.length - enriched,
        pending,
        latencyMs,
        provider: provider.name,
      });

      return {
        success: true,
        enriched,
        requested: batch.length,
        unparsed: batch.length - enriched,
        pending,
        latencyMs,
      };
    },
  );
}
