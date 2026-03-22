"use client";

import type { SatChangeManifest } from "@/lib/contracts";

type JobDetailsProps = {
  manifest: SatChangeManifest;
  isInSidebar?: boolean;
};

type MetadataItem = {
  label: string;
  value: string;
  title?: string;
};

export function JobDetails({ manifest, isInSidebar = false }: JobDetailsProps) {
  const metadata: MetadataItem[] = [
    { label: "Job ID", value: manifest.job_id, title: manifest.job_id },
    {
      label: "Output Prefix",
      value: manifest.output_prefix,
      title: manifest.output_prefix,
    },
    {
      label: "Coordinates",
      value: `${manifest.center.lat.toFixed(5)}, ${manifest.center.lon.toFixed(5)}`,
    },
    {
      label: "Dimensions",
      value: `${manifest.dimensions.width ?? "—"} × ${manifest.dimensions.height ?? "—"} px`,
    },
  ];

  return (
    <div className={isInSidebar ? "space-y-4" : "data-panel rounded p-4 space-y-4"}>
      <section className="rounded-md border border-border/70 bg-muted/10 p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-3">
          Metadata
        </p>
        <div className="grid gap-2 sm:grid-cols-2 text-xs">
          {metadata.map((item) => (
            <article
              key={item.label}
              className="rounded border border-border/60 bg-muted/20 px-2.5 py-2"
            >
              <p className="text-[10px] font-mono text-muted-foreground uppercase mb-1">
                {item.label}
              </p>
              <p className="font-mono text-foreground/85 truncate" title={item.title}>
                {item.value}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-md border border-border/70 bg-muted/10 p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-3">
          Artifacts
        </p>
        <div className="space-y-1.5">
          {Object.entries(manifest.artifacts).map(([key, value]) => (
            <article key={key} className="rounded border border-border/60 bg-muted/20 px-2.5 py-2">
              <p className="text-[10px] font-mono text-muted-foreground uppercase">
                {formatArtifactKey(key)}
              </p>
              <p className="mt-1 text-xs font-mono text-foreground/85 truncate" title={value || "—"}>
                {value ? extractFileName(value) : "—"}
              </p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function formatArtifactKey(key: string): string {
  return key.replace(/_/g, " ");
}

function extractFileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] ?? path;
}
