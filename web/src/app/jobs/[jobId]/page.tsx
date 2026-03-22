import { notFound } from "next/navigation";

import { JobViewer } from "@/components/job-viewer";
import { loadManifestByJobId } from "@/lib/job-manifest";
import type { SatChangeManifest } from "@/lib/contracts";

type JobPageProps = {
  params: { jobId: string };
};

async function loadManifest(jobId: string): Promise<SatChangeManifest> {
  const baseDir = process.env.SATCHANGE_RESULTS_DIR;
  if (!baseDir) {
    throw new Error("SATCHANGE_RESULTS_DIR is not configured");
  }
  return loadManifestByJobId(baseDir, jobId);
}

export default async function JobPage({ params }: JobPageProps) {
  let manifest: SatChangeManifest;
  try {
    manifest = await loadManifest(params.jobId);
  } catch {
    notFound();
  }

  return <JobViewer manifest={manifest} jobId={params.jobId} />;
}

