export default function HomePage() {
  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-md w-full text-center">
        {/* Title */}
        <h1 className="text-3xl font-semibold tracking-tight mb-2">
          <span className="text-primary">Sat</span>Change
        </h1>
        <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-8">
          Earth Observation Platform
        </p>
        
        {/* Description */}
        <p className="text-sm text-muted-foreground mb-8">
          Visualize satellite imagery change detection results.
        </p>
        
        {/* Instructions */}
        <div className="data-panel rounded p-5 text-left">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-wide mb-3">
            Quick Start
          </p>
          
          <p className="text-sm text-muted-foreground mb-3">
            Navigate to a job viewer using the Job ID:
          </p>
          
          <div className="bg-muted/40 rounded p-3 font-mono text-sm">
            <span className="text-muted-foreground">/jobs/</span>
            <span className="text-primary">&lt;job_id&gt;</span>
          </div>
          
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-xs text-muted-foreground mb-2">
              Run an analysis to generate a Job ID:
            </p>
            <code className="block text-xs font-mono text-foreground/70 bg-muted/30 px-3 py-2 rounded">
              python -m satchange analyze --center "13.08,80.27" --date-a "2022-01-01" --date-b "2024-01-01" --output .\results
            </code>
          </div>
        </div>
        
        {/* Footer */}
        <p className="mt-6 text-[10px] font-mono text-muted-foreground/50 uppercase tracking-wide">
          Sentinel-2 • Google Earth Engine
        </p>
      </div>
    </main>
  );
}

