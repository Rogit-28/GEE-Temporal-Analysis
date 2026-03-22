import type { ViewerStatCard } from "@/lib/contracts";

type StatsGridProps = {
  cards: ViewerStatCard[];
};

// Semantic colors for data visualization (kept intentionally)
const STAT_COLORS: Record<string, string> = {
  total_area_changed: "text-primary",
  vegetation_growth: "text-green-400",
  vegetation_loss: "text-red-400",
  water_expansion: "text-blue-400",
  water_reduction: "text-orange-400",
  urban_development: "text-slate-300",
  urban_decline: "text-slate-400",
  changed_area: "text-primary",
};

export function StatsGrid({ cards }: StatsGridProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 stagger-fade-in">
      {cards.map((card) => {
        const colorClass = STAT_COLORS[card.key] ?? "text-foreground";
        
        return (
          <div 
            key={card.key} 
            className="data-panel rounded p-3 hover:border-primary/40 transition-colors"
          >
            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wide mb-1 truncate">
              {card.label}
            </p>
            <p className={`text-lg font-semibold font-mono ${colorClass}`}>
              {card.value}
            </p>
          </div>
        );
      })}
    </div>
  );
}
