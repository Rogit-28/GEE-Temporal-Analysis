import { NextResponse } from "next/server";

import { hasViewerPayload } from "../../../../lib/contracts";
import { loadManifestByJobId } from "../../../../lib/job-manifest";

type Params = { params: { jobId: string } };

export async function GET(_: Request, { params }: Params) {
  const baseDir = process.env.SATCHANGE_RESULTS_DIR;
  if (!baseDir) {
    return NextResponse.json(
      { error: "SATCHANGE_RESULTS_DIR is not configured" },
      { status: 500 }
    );
  }

  try {
    const manifest = await loadManifestByJobId(baseDir, params.jobId);
    if (!hasViewerPayload(manifest)) {
      return NextResponse.json(
        {
          error:
            "Manifest missing viewer payload. Re-run export with latest SatChange.",
        },
        { status: 422 }
      );
    }
    return NextResponse.json(manifest);
  } catch (e) {
    if (e instanceof Error && e.message === "Job not found") {
      return NextResponse.json({ error: "Job not found" }, { status: 404 });
    }
    if (e instanceof Error && e.message === "Invalid job id") {
      return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
    }
    if (e instanceof Error && e.message.includes("Path escapes base directory")) {
      return NextResponse.json({ error: "Invalid job path" }, { status: 400 });
    }
    console.error("Failed to resolve manifest", e);
    return NextResponse.json(
      { error: "Failed to resolve manifest" },
      { status: 500 }
    );
  }
}

