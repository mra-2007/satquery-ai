import { useEffect, useState } from "react";

import { ApiError, getScenes, getSceneChangeSummary, getSceneLegend, postQuery } from "./api/client";
import type { ChangeSummary, LegendEntry, QueryResponse, SceneSummary, UploadResponse } from "./api/types";
import AnswerPanel from "./components/AnswerPanel";
import ChangeView from "./components/ChangeView";
import CommandBar, { ANALYZE_EXAMPLES, CHANGE_EXAMPLES, CROSS_MODAL_EXAMPLES } from "./components/CommandBar";
import CrossModalView from "./components/CrossModalView";
import GainLossTable from "./components/GainLossTable";
import Legend from "./components/Legend";
import MapView from "./components/MapView";
import TopBar from "./components/TopBar";
import TraceDrawer from "./components/TraceDrawer";
import VerificationPanel from "./components/VerificationPanel";
import WarningBanner from "./components/WarningBanner";
import { PREFERRED_DEFAULT_SCENE_ID } from "./utils/scene";

export default function App() {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [maskOpacity, setMaskOpacity] = useState(0.65);
  const [legend, setLegend] = useState<LegendEntry[]>([]);
  const [changeSummary, setChangeSummary] = useState<ChangeSummary | null>(null);
  const [changeSummaryLoading, setChangeSummaryLoading] = useState(false);
  const [changeSummaryError, setChangeSummaryError] = useState<string | null>(null);

  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [verificationOpen, setVerificationOpen] = useState(false);

  const [scenesError, setScenesError] = useState<string | null>(null);

  function refreshScenes(selectAfter?: string) {
    return getScenes()
      .then((loaded) => {
        setScenes(loaded);
        if (selectAfter) {
          setSelectedSceneId(selectAfter);
        } else if (loaded.length > 0 && !selectedSceneId) {
          const preferred = loaded.find((s) => s.id === PREFERRED_DEFAULT_SCENE_ID);
          setSelectedSceneId((preferred ?? loaded[0]).id);
        }
      })
      .catch((err) => setScenesError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refreshScenes();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedScene = scenes.find((s) => s.id === selectedSceneId) ?? null;
  const isChangeScene = selectedScene?.kind === "change";
  const isCrossModalScene = selectedScene?.kind === "cross_modal";

  useEffect(() => {
    if (!selectedSceneId) return;
    // clear any answer from the previous scene -- it's no longer relevant
    setQueryResponse(null);
    setQueryError(null);
    setTraceOpen(false);
    setVerificationOpen(false);

    if (isChangeScene) {
      setLegend([]);
      setChangeSummary(null);
      setChangeSummaryError(null);
      setChangeSummaryLoading(true);
      getSceneChangeSummary(selectedSceneId)
        .then(setChangeSummary)
        .catch((err) => setChangeSummaryError(err instanceof Error ? err.message : String(err)))
        .finally(() => setChangeSummaryLoading(false));
    } else if (isCrossModalScene) {
      // CrossModalView owns this visual slot itself (its own findings
      // panel, fetched from GET /scenes/{id}/cross-modal) -- neither the
      // single-mask Legend nor the bi-temporal GainLossTable applies here.
      setLegend([]);
      setChangeSummary(null);
      setChangeSummaryError(null);
    } else {
      setChangeSummary(null);
      setChangeSummaryError(null);
      getSceneLegend(selectedSceneId)
        .then(setLegend)
        .catch(() => setLegend([]));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSceneId, isChangeScene, isCrossModalScene]);

  async function handleQuery(query: string) {
    if (!selectedSceneId) return;
    setQueryLoading(true);
    setQueryError(null);
    try {
      const response = await postQuery(selectedSceneId, query);
      setQueryResponse(response);
    } catch (err) {
      setQueryResponse(null);
      setQueryError(err instanceof ApiError ? err.message : "The scene could not be analyzed.");
    } finally {
      setQueryLoading(false);
    }
  }

  function handleUploaded(response: UploadResponse) {
    if (response.scene_id) refreshScenes(response.scene_id);
  }

  return (
    <div className="app-shell">
      <TopBar
        scenes={scenes}
        selectedScene={selectedScene}
        onSelectScene={setSelectedSceneId}
        maskOpacity={maskOpacity}
        onMaskOpacityChange={setMaskOpacity}
        onUploaded={handleUploaded}
      />

      <WarningBanner warnings={selectedScene?.warnings ?? []} />

      <main className="app-main">
        {selectedScene ? (
          isChangeScene ? (
            <ChangeView
              sceneId={selectedScene.id}
              width={selectedScene.width}
              height={selectedScene.height}
              gsdMetres={selectedScene.gsd_metres}
              beforeDate={selectedScene.before_date}
              afterDate={selectedScene.after_date}
            />
          ) : isCrossModalScene ? (
            <CrossModalView
              sceneId={selectedScene.id}
              width={selectedScene.width}
              height={selectedScene.height}
              gsdMetres={selectedScene.gsd_metres}
            />
          ) : (
            <MapView
              sceneId={selectedScene.id}
              width={selectedScene.width}
              height={selectedScene.height}
              gsdMetres={selectedScene.gsd_metres}
              maskOpacity={maskOpacity}
            />
          )
        ) : (
          <div className="app-shell__loading">
            {scenesError ? `Could not reach the backend: ${scenesError}` : "Loading scenes…"}
          </div>
        )}

        {isChangeScene ? (
          <GainLossTable summary={changeSummary} loading={changeSummaryLoading} error={changeSummaryError} />
        ) : isCrossModalScene ? null : (
          <Legend entries={legend} />
        )}

        <AnswerPanel
          response={queryResponse}
          loading={queryLoading}
          error={queryError}
          traceOpen={traceOpen}
          onToggleTrace={() => setTraceOpen((open) => !open)}
          verificationOpen={verificationOpen}
          onToggleVerification={() => setVerificationOpen((open) => !open)}
        />

        <TraceDrawer
          trace={queryResponse ? queryResponse.evidence.execution_trace : null}
          open={traceOpen && queryResponse != null}
          onClose={() => setTraceOpen(false)}
        />

        <VerificationPanel
          trace={queryResponse ? queryResponse.evidence.execution_trace : null}
          open={verificationOpen && queryResponse != null}
          onClose={() => setVerificationOpen(false)}
        />

        <CommandBar
          onSubmit={handleQuery}
          loading={queryLoading}
          examples={isChangeScene ? CHANGE_EXAMPLES : isCrossModalScene ? CROSS_MODAL_EXAMPLES : ANALYZE_EXAMPLES}
        />
      </main>
    </div>
  );
}
