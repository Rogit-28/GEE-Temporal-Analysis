import { promises as fs } from "node:fs";
import path from "node:path";

import { parseManifest, type SatChangeManifest } from "./contracts";

type JobIndex = {
  job_id?: string;
  manifest_path?: string;
};

const SAFE_JOB_ID_RE = /^[A-Za-z0-9._-]+$/;

function assertSafeJobId(jobId: string): void {
  if (!SAFE_JOB_ID_RE.test(jobId)) {
    throw new Error("Invalid job id");
  }
}

function resolveWithinBase(baseDir: string, targetPath: string): string {
  const baseResolved = path.resolve(baseDir);
  const candidate = path.resolve(baseResolved, targetPath);
  if (!candidate.startsWith(baseResolved + path.sep) && candidate !== baseResolved) {
    throw new Error("Path escapes base directory");
  }
  return candidate;
}

function resolveManifestPath(baseDir: string, manifestPath: string): string {
  if (path.isAbsolute(manifestPath)) {
    return resolveWithinBase(baseDir, path.relative(baseDir, manifestPath));
  }
  return resolveWithinBase(baseDir, manifestPath);
}

export async function loadManifestByJobId(
  baseDir: string,
  jobId: string
): Promise<SatChangeManifest> {
  assertSafeJobId(jobId);
  const safeBaseDir = path.resolve(baseDir);

  // First, try modern bundle structure: {jobId}_web_bundle/manifest.json
  try {
    const bundleDir = resolveWithinBase(safeBaseDir, `${jobId}_web_bundle`);
    const manifestPath = resolveWithinBase(bundleDir, "manifest.json");
    const manifestRaw = await fs.readFile(manifestPath, "utf-8");
    return parseManifest(JSON.parse(manifestRaw));
  } catch {
    // Fall back to legacy _job.json index files
  }

  const files = await fs.readdir(safeBaseDir);
  const jobIndexFiles = files.filter((file) => file.endsWith("_job.json"));

  for (const indexFile of jobIndexFiles) {
    const indexPath = resolveWithinBase(safeBaseDir, indexFile);
    const raw = await fs.readFile(indexPath, "utf-8");
    const index = JSON.parse(raw) as JobIndex;
    if (index.job_id !== jobId || !index.manifest_path) {
      continue;
    }

    const manifestPath = resolveManifestPath(safeBaseDir, index.manifest_path);
    const manifestRaw = await fs.readFile(manifestPath, "utf-8");
    return parseManifest(JSON.parse(manifestRaw));
  }

  throw new Error("Job not found");
}

