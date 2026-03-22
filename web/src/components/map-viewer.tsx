"use client";

import { useEffect, useRef, useState } from "react";
import type { Map as LeafletMap, ImageOverlay } from "leaflet";
import type {
  LayerName,
  SatChangeViewerPayload,
  ViewerLegendItem,
} from "@/lib/contracts";

type MapViewerProps = {
  viewer: SatChangeViewerPayload;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
};

const LAYER_CONFIG: Record<LayerName, { label: string; shortLabel: string }> = {
  before: { label: "Before", shortLabel: "A" },
  after: { label: "After", shortLabel: "B" },
  changes: { label: "Changes", shortLabel: "Δ" },
};

function LegendOverlay({ items, isVisible }: { items: ViewerLegendItem[]; isVisible: boolean }) {
  if (!isVisible) return null;

  return (
    <div className="absolute bottom-4 right-4 z-[30] bg-card/95 backdrop-blur-sm border border-border rounded p-3 min-w-[150px]">
      <h4 className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider mb-2 pb-1.5 border-b border-border">
        Classification
      </h4>
      <div className="space-y-1">
        {items.map((item) => (
          <div key={item.id} className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-sm"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-[10px] text-foreground/80">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CoordinatesDisplay({ lat, lon }: { lat: number; lon: number }) {
  return (
    <div className="absolute bottom-4 left-4 z-[1000] px-2 py-1 bg-card/90 backdrop-blur-sm border border-border rounded">
      <div className="flex items-center gap-2 font-mono text-[9px] text-muted-foreground">
        <span>{lat.toFixed(5)}°</span>
        <span className="text-border/60">|</span>
        <span>{lon.toFixed(5)}°</span>
      </div>
    </div>
  );
}

export function MapViewer({ viewer, isFullscreen = false, onToggleFullscreen }: MapViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const layersRef = useRef<Record<LayerName, ImageOverlay> | null>(null);
  const [activeLayer, setActiveLayer] = useState<LayerName>(viewer.controls.default_layer);
  const [mapError, setMapError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    async function initMap() {
      if (!containerRef.current || mapRef.current) return;

      try {
        setIsLoading(true);
        const leaflet = await import("leaflet");
        if (!isMounted || !containerRef.current) return;

        const map = leaflet.map(containerRef.current, {
          zoomControl: false,
          attributionControl: true,
        }).setView([viewer.map.center.lat, viewer.map.center.lon], viewer.map.zoom);
        mapRef.current = map;

        leaflet
          .tileLayer(viewer.map.tile_url, {
            attribution: viewer.map.tile_attribution,
            maxZoom: 19,
          })
          .addTo(map);

        leaflet.control.zoom({ position: "topright" }).addTo(map);

        const bounds = leaflet.latLngBounds(viewer.map.bounds[0], viewer.map.bounds[1]);

        const before = leaflet.imageOverlay(viewer.layers.before.image_uri, bounds, { opacity: 1 });
        const after = leaflet.imageOverlay(viewer.layers.after.image_uri, bounds, { opacity: 1 });
        const changes = leaflet.imageOverlay(viewer.layers.changes.image_uri, bounds, { opacity: 1 });
        layersRef.current = { before, after, changes };

        layersRef.current[viewer.controls.default_layer].addTo(map);
        map.fitBounds(bounds, { padding: [30, 30] });

        if (isMounted) setIsLoading(false);
      } catch (error) {
        if (isMounted) {
          setMapError(error instanceof Error ? error.message : "Failed to initialize map");
          setIsLoading(false);
        }
      }
    }

    initMap();

    return () => {
      isMounted = false;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      layersRef.current = null;
    };
  }, [viewer]);

  // Invalidate size when fullscreen changes
  useEffect(() => {
    if (mapRef.current) {
      setTimeout(() => mapRef.current?.invalidateSize(), 100);
    }
  }, [isFullscreen]);

  useEffect(() => {
    const map = mapRef.current;
    const layers = layersRef.current;
    if (!map || !layers) return;

    (Object.keys(layers) as LayerName[]).forEach((name) => {
      const layer = layers[name];
      if (map.hasLayer(layer)) map.removeLayer(layer);
    });
    layers[activeLayer].addTo(map);
  }, [activeLayer]);

  const beforeDate = viewer.layers.before.date ?? "—";
  const afterDate = viewer.layers.after.date ?? "—";

  return (
    <div className={`flex flex-col bg-card border border-border rounded overflow-hidden ${isFullscreen ? "h-full" : "h-full"}`}>
      {/* Header: dates left, toggles right */}
      <div className="shrink-0 px-3 py-2 border-b border-border flex items-center justify-between bg-card relative">
        {/* Subtle accent line */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

        {/* Left: Date range */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-muted-foreground">{beforeDate}</span>
          <svg className="w-3 h-3 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
          <span className="text-muted-foreground">{afterDate}</span>
        </div>

        {/* Right: Layer toggles + fullscreen */}
        <div className="flex items-center gap-2">
          {/* Layer toggles */}
          <div className="flex items-center bg-muted/30 rounded p-0.5">
            {(["before", "after", "changes"] as const).map((layer) => {
              const config = LAYER_CONFIG[layer];
              const isActive = activeLayer === layer;
              return (
                <button
                  key={layer}
                  onClick={() => setActiveLayer(layer)}
                  className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                    isActive
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {config.label}
                </button>
              );
            })}
          </div>

          {/* Fullscreen toggle */}
          {onToggleFullscreen && (
            <button
              onClick={onToggleFullscreen}
              className="p-1.5 rounded hover:bg-muted/50 text-muted-foreground hover:text-foreground transition-colors"
              title={isFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen (F)"}
            >
              {isFullscreen ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9L4 4m0 0v5m0-5h5m6 6l5 5m0 0v-5m0 5h-5M9 15l-5 5m0 0v-5m0 5h5m6-6l5-5m0 0v5m0-5h-5" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-5h-4m4 0v4m0-4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5h-4m4 0v-4m0 4l-5-5" />
                </svg>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Map container - fills remaining space */}
      <div className="flex-1 relative min-h-0">
        {mapError ? (
          <div className="absolute inset-0 flex items-center justify-center bg-destructive/5">
            <div className="text-center p-4">
              <p className="text-destructive font-medium text-sm mb-1">Failed to load</p>
              <p className="text-xs text-muted-foreground font-mono">{mapError}</p>
            </div>
          </div>
        ) : isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        ) : null}

        <div
          ref={containerRef}
          className="w-full h-full"
          style={{ visibility: isLoading || mapError ? "hidden" : "visible" }}
        />

        {!mapError && !isLoading && (
          <>
            <LegendOverlay items={viewer.legend} isVisible={activeLayer === "changes"} />
            <CoordinatesDisplay lat={viewer.map.center.lat} lon={viewer.map.center.lon} />
          </>
        )}
      </div>
    </div>
  );
}
