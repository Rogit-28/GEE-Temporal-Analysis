"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { JobDetails } from "@/components/job-details";
import { StatsGrid } from "@/components/stats-grid";
import { hasViewerPayload, type SatChangeManifest } from "@/lib/contracts";

const MapViewer = dynamic(
  () => import("@/components/map-viewer").then((mod) => mod.MapViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center bg-card border border-border rounded">
        <div className="text-center">
          <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="mt-2 text-xs font-mono text-muted-foreground">Loading...</p>
        </div>
      </div>
    ),
  }
);

type JobViewerProps = {
  manifest: SatChangeManifest;
  jobId: string;
};

export function JobViewer({ manifest, jobId }: JobViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const hasViewer = hasViewerPayload(manifest);
  const coords = `${manifest.center.lat.toFixed(4)}°, ${manifest.center.lon.toFixed(4)}°`;

  // Keyboard shortcut for fullscreen
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "Escape" && isFullscreen) {
      setIsFullscreen(false);
    }
    if (e.key === "f" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const target = e.target as HTMLElement;
      if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
        setIsFullscreen((prev) => !prev);
      }
    }
  }, [isFullscreen]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Calculate stats summary
  const statsSummary = hasViewer
    ? `${manifest.viewer!.stats_cards.length} metrics`
    : "No data";

  if (!hasViewer) {
    return (
      <div className="h-screen flex flex-col">
        <Navbar coords={coords} jobId={jobId} />
        <div className="flex-1 flex items-center justify-center">
          <div className="data-panel rounded p-6 max-w-md text-center">
            <p className="text-destructive font-medium mb-2">Viewer data not available</p>
            <p className="text-sm text-muted-foreground">
              Re-run export with the latest version to generate viewer payload.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Fullscreen mode
  if (isFullscreen) {
    return (
      <div className="h-screen flex flex-col bg-background">
        <MapViewer
          viewer={manifest.viewer!}
          isFullscreen={true}
          onToggleFullscreen={() => setIsFullscreen(false)}
        />
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Navbar */}
      <Navbar coords={coords} jobId={jobId} onShowDetails={() => setShowDetails(true)} />

      {/* Main content - viewport fit */}
      <div className="flex-1 flex flex-col min-h-0 p-3 gap-3">
        {/* Map - fills available space */}
        <div className="flex-1 min-h-0">
          <MapViewer
            viewer={manifest.viewer!}
            isFullscreen={false}
            onToggleFullscreen={() => setIsFullscreen(true)}
          />
        </div>

        {/* Stats bar - collapsed by default */}
        <div className="shrink-0">
          <button
            onClick={() => setShowStats(!showStats)}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-card border border-border rounded hover:border-primary/40 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground uppercase tracking-wide">
                Change Metrics
              </span>
              <span className="text-xs text-primary font-mono">{statsSummary}</span>
            </div>
            <svg
              className={`w-4 h-4 text-muted-foreground transition-transform ${showStats ? "rotate-180" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Expanded stats */}
          <div
            className={`overflow-hidden transition-all duration-300 ease-out ${
              showStats ? "max-h-[300px] mt-3" : "max-h-0"
            }`}
          >
            <StatsGrid cards={manifest.viewer!.stats_cards} />
          </div>
        </div>
      </div>

      {/* Job details modal/drawer */}
      {showDetails && (
        <div
          className="fixed inset-0 z-40 bg-background/78 backdrop-blur-sm"
          onClick={() => setShowDetails(false)}
        >
          <div
            className="absolute right-0 top-0 z-50 h-full w-full max-w-lg overflow-auto border-l border-border/70 bg-card/95 shadow-[0_24px_60px_rgba(0,0,0,0.45)] backdrop-blur-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 z-10 border-b border-border/70 bg-card/95 px-4 py-3 backdrop-blur-md">
              <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_82%_0%,hsl(var(--primary)/0.08),transparent_45%)]" />
              <div className="relative flex items-center justify-between">
                <div className="min-w-0">
                  <p className="text-[10px] font-mono uppercase tracking-[0.14em] text-muted-foreground">
                    Job Details
                  </p>
                  <h2 className="mt-1 truncate font-mono text-xs text-foreground/90" title={jobId}>
                    {jobId}
                  </h2>
                </div>
              <button
                onClick={() => setShowDetails(false)}
                className="rounded border border-border/70 bg-muted/40 p-1.5 text-muted-foreground transition-colors hover:border-primary/35 hover:text-primary"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              </div>
            </div>
            <div className="p-4">
              <JobDetails manifest={manifest} isInSidebar={true} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Polished navbar component
function Navbar({
  coords,
  jobId,
  onShowDetails,
}: {
  coords: string;
  jobId: string;
  onShowDetails?: () => void;
}) {
  return (
    <header className="shrink-0 relative border-b border-border bg-card/95 backdrop-blur-md">
      {/* Subtle accent line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_15%_10%,hsl(var(--primary)/0.06),transparent_35%)]" />

      <div className="relative px-4 py-2.5 flex items-center justify-between gap-3">
        {/* Branding */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded bg-primary/10 border border-primary/25 shadow-[0_0_0_1px_hsl(var(--primary)/0.08)] flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-4 h-4 text-primary" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <span className="text-base font-semibold tracking-tight">
              <span className="text-primary">Sat</span>Change
            </span>
          </div>
        </div>

        {/* Center: coordinates */}
        <div className="absolute left-1/2 -translate-x-1/2 hidden md:block">
          <span className="text-xs font-mono text-muted-foreground">{coords}</span>
        </div>

        {/* Right: job details button */}
        <button
          onClick={onShowDetails}
          className="group inline-flex items-center gap-2 rounded-md border border-border/70 bg-muted/40 px-2.5 py-1.5 shadow-sm transition-all hover:border-primary/45 hover:bg-primary/5 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.16)]"
          title={jobId}
        >
          <span className="inline-flex items-center rounded border border-border/70 bg-card/70 px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wider text-muted-foreground group-hover:text-primary transition-colors">
            Job Details
          </span>
          <code className="max-w-[26ch] truncate text-xs font-mono text-foreground/80 group-hover:text-foreground transition-colors">
            {jobId}
          </code>
          <svg
            viewBox="0 0 20 20"
            className="h-3.5 w-3.5 text-muted-foreground/70 transition-transform group-hover:translate-x-0.5 group-hover:text-primary"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M7 5l5 5-5 5" />
          </svg>
        </button>
      </div>
    </header>
  );
}
