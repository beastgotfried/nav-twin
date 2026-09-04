import { useEffect, useState } from "react";

/**
 * Integration status, polled from GET /api/bridge/status. This card is the
 * visible end of the production bridge: external systems push telemetry
 * through POST /api/ingest (native schema or MAVLink/EFI envelope) or an
 * MQTT broker, and those frames run through the same twin, the same bands
 * and the same diagnosis as any simulated mission.
 */

interface BridgeStatus {
  listening: boolean;
  session: string | null;
  frames_ingested: number;
  frames_rejected: number;
  sources: Record<string, number>;
  queue_depth: number;
  last_frame_s_ago: number | null;
  endpoints: Record<string, string>;
}

export function IntegrationCard() {
  const [st, setSt] = useState<BridgeStatus | null>(null);

  useEffect(() => {
    let dead = false;
    const poll = async () => {
      try {
        const res = await fetch("/api/bridge/status");
        if (res.ok && !dead) setSt(await res.json());
      } catch {
        /* server without the bridge endpoints: card stays quiet */
      }
    };
    void poll();
    const id = window.setInterval(poll, 3000);
    return () => {
      dead = true;
      window.clearInterval(id);
    };
  }, []);

  const active = st !== null && st.session === "external";
  const seen =
    st?.last_frame_s_ago !== null && st?.last_frame_s_ago !== undefined
      ? `${st.last_frame_s_ago.toFixed(0)} s ago`
      : "never";

  return (
    <section className={`card ${active ? "alert-caution" : ""}`}>
      <h2 className="card-title">INTEGRATION</h2>
      <p className="alert-sub">external telemetry bridge</p>
      {st === null ? (
        <p className="empty-hint">bridge status unavailable</p>
      ) : (
        <>
          <div className="kv-row">
            <span className="kv-label">feed</span>
            <span className="kv-value mono">
              {active ? "EXTERNAL LIVE" : "listening"}
            </span>
          </div>
          <div className="kv-row">
            <span className="kv-label">frames ingested</span>
            <span className="kv-value mono">{st.frames_ingested}</span>
          </div>
          <div className="kv-row">
            <span className="kv-label">last frame</span>
            <span className="kv-value mono">{seen}</span>
          </div>
          <div className="kv-row">
            <span className="kv-label">rejected</span>
            <span className="kv-value mono">{st.frames_rejected}</span>
          </div>
          <p className="adv-foot">
            POST /api/ingest · native or MAVLink/EFI · MQTT via NAVTWIN_MQTT
          </p>
        </>
      )}
    </section>
  );
}
