export type LayerName = "before" | "after" | "changes";

export interface ViewerLayer {
  label: string;
  image_uri: string;
  date?: string;
}

export interface ViewerLegendItem {
  id: string;
  label: string;
  color: string;
}

export interface ViewerStatCard {
  key: string;
  label: string;
  value: string;
}

export interface ViewerMapConfig {
  center: {
    lat: number;
    lon: number;
  };
  zoom: number;
  bounds: [[number, number], [number, number]];
  tile_url: string;
  tile_attribution: string;
}

export interface SatChangeViewerPayload {
  map: ViewerMapConfig;
  controls: {
    default_layer: LayerName;
  };
  layers: Record<LayerName, ViewerLayer>;
  legend: ViewerLegendItem[];
  stats_cards: ViewerStatCard[];
}

export interface SatChangeManifest {
  schema_version: string;
  job_id: string;
  output_prefix: string;
  created_at: string;
  center: {
    lat: number;
    lon: number;
  };
  dimensions: {
    height: number | null;
    width: number | null;
  };
  artifacts: Record<string, string | null>;
  stats: Record<string, unknown>;
  metadata: Record<string, unknown>;
  class_mapping: Record<string, string>;
  viewer?: SatChangeViewerPayload;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isLayerName(value: unknown): value is LayerName {
  return value === "before" || value === "after" || value === "changes";
}

export function parseManifest(input: unknown): SatChangeManifest {
  if (!input || typeof input !== "object") {
    throw new Error("Invalid manifest payload");
  }
  const manifest = input as Partial<SatChangeManifest>;
  if (
    typeof manifest.job_id !== "string" ||
    typeof manifest.schema_version !== "string" ||
    typeof manifest.output_prefix !== "string" ||
    !manifest.artifacts ||
    typeof manifest.artifacts !== "object" ||
    !manifest.center ||
    !isFiniteNumber(manifest.center.lat) ||
    !isFiniteNumber(manifest.center.lon)
  ) {
    throw new Error("Manifest is missing required fields");
  }
  return manifest as SatChangeManifest;
}

export function hasViewerPayload(
  manifest: SatChangeManifest
): manifest is SatChangeManifest & { viewer: SatChangeViewerPayload } {
  const viewer = manifest.viewer;
  if (!viewer) {
    return false;
  }
  return (
    !!viewer.map &&
    isFiniteNumber(viewer.map.center?.lat) &&
    isFiniteNumber(viewer.map.center?.lon) &&
    isFiniteNumber(viewer.map.zoom) &&
    Array.isArray(viewer.map.bounds) &&
    typeof viewer.map.tile_url === "string" &&
    typeof viewer.map.tile_attribution === "string" &&
    !!viewer.layers &&
    !!viewer.controls &&
    isLayerName(viewer.controls.default_layer) &&
    Object.values(viewer.layers).every(
      (layer) => !!layer && typeof layer.label === "string" && typeof layer.image_uri === "string"
    ) &&
    Array.isArray(viewer.legend) &&
    Array.isArray(viewer.stats_cards)
  );
}

