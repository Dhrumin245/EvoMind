import { Download, Eye, GitCompareArrows } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useGenomeDetail, useGenomes } from '../../api/hooks';
import type { GenomeSummary } from '../../api/types';
import { formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';

function downloadGenomeSummary(genome: GenomeSummary) {
  const blob = new Blob([JSON.stringify(genome, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${genome.genome_type}-${genome.genome_id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function GenomePanel({ title, genomes }: { title: string; genomes: GenomeSummary[] }) {
  const best = genomes[0];
  return (
    <div className="rounded-md border border-border bg-card p-4 shadow-panel">
      <h2 className="text-base font-semibold">{title}</h2>
      {best ? (
        <div className="mt-4 space-y-4">
          <dl className="grid gap-3 text-sm sm:grid-cols-2">
            <div><dt className="text-muted-foreground">Fitness</dt><dd className="font-medium">{formatNumber(best.fitness)}</dd></div>
            <div><dt className="text-muted-foreground">Generation</dt><dd className="font-medium">{best.generation}</dd></div>
            <div><dt className="text-muted-foreground">Genes</dt><dd className="font-medium">{best.gene_count}</dd></div>
            <div><dt className="text-muted-foreground">Architecture</dt><dd className="font-medium">{best.architecture || 'Unknown'}</dd></div>
            <div><dt className="text-muted-foreground">Input size</dt><dd className="font-medium">{best.input_size}</dd></div>
            <div><dt className="text-muted-foreground">Output size</dt><dd className="font-medium">{best.output_size}</dd></div>
          </dl>
          <div className="flex flex-wrap gap-2">
            <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={() => downloadGenomeSummary(best)} type="button">
              <Download className="h-4 w-4" aria-hidden="true" />
              JSON
            </button>
            <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm disabled:opacity-50" disabled type="button">
              <Download className="h-4 w-4" aria-hidden="true" />
              PyTorch
            </button>
          </div>
        </div>
      ) : (
        <p className="mt-8 text-sm text-muted-foreground">No genomes found.</p>
      )}
    </div>
  );
}

export function GenomeManager() {
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const genomes = useGenomes(selectedJobId);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [detailGenome, setDetailGenome] = useState<GenomeSummary | null>(null);
  const items = genomes.data?.items || [];
  const prey = useMemo(() => items.filter((item) => item.genome_type === 'prey').sort((a, b) => b.fitness - a.fitness), [items]);
  const predator = useMemo(() => items.filter((item) => item.genome_type === 'predator').sort((a, b) => b.fitness - a.fitness), [items]);
  const compareItems = compareIds.map((id) => items.find((item) => item.genome_id === id)).filter(Boolean) as GenomeSummary[];
  const genomeDetail = useGenomeDetail(selectedJobId, detailGenome?.genome_id, detailGenome?.genome_type);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Genomes</h1>
          <p className="mt-1 text-sm text-muted-foreground">Best prey and predator genomes for the selected job.</p>
        </div>
        <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={() => setCompareOpen(true)} type="button">
          <GitCompareArrows className="h-4 w-4" aria-hidden="true" />
          Compare
        </button>
      </div>

      <section className="grid gap-4 xl:grid-cols-2">
        <GenomePanel title="Prey Best Genome" genomes={prey} />
        <GenomePanel title="Predator Best Genome" genomes={predator} />
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-panel">
        <h2 className="text-base font-semibold">Genome Catalog</h2>
        <div className="mt-4 overflow-auto table-scroll">
          <table className="w-full min-w-[780px] text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3">Select</th>
                <th className="py-2 pr-3">Genome</th>
                <th className="py-2 pr-3">Type</th>
                <th className="py-2 pr-3">Fitness</th>
                <th className="py-2 pr-3">Generation</th>
                <th className="py-2 pr-3">Genes</th>
                <th className="py-2 pr-3">Source</th>
                <th className="py-2 pr-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((genome) => (
                <tr className="border-b border-border last:border-0" key={genome.genome_id}>
                  <td className="py-3 pr-3">
                    <input
                      checked={compareIds.includes(genome.genome_id)}
                      onChange={(event) => {
                        setCompareIds((current) => {
                          if (event.target.checked) {
                            return [...current, genome.genome_id].slice(-2);
                          }
                          return current.filter((id) => id !== genome.genome_id);
                        });
                      }}
                      type="checkbox"
                    />
                  </td>
                  <td className="py-3 pr-3 font-medium">{genome.genome_id}</td>
                  <td className="py-3 pr-3 capitalize">{genome.genome_type}</td>
                  <td className="py-3 pr-3">{formatNumber(genome.fitness)}</td>
                  <td className="py-3 pr-3">{genome.generation}</td>
                  <td className="py-3 pr-3">{genome.gene_count}</td>
                  <td className="py-3 pr-3 text-muted-foreground">{genome.source}</td>
                  <td className="py-3 pr-3">
                    <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setDetailGenome(genome)} type="button" title="Details">
                      <Eye className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!items.length && <p className="py-8 text-center text-sm text-muted-foreground">No genomes available.</p>}
        </div>
      </section>

      {compareOpen && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-slate-950/35 p-4">
          <div className="w-full max-w-3xl rounded-md border border-border bg-card p-5 shadow-xl">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Genome Compare</h2>
              <button className="rounded-md border border-border px-3 py-2 text-sm" onClick={() => setCompareOpen(false)} type="button">Close</button>
            </div>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              {compareItems.map((genome) => (
                <pre className="max-h-96 overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-50" key={genome.genome_id}>
                  {JSON.stringify(genome, null, 2)}
                </pre>
              ))}
            </div>
            {compareItems.length < 2 && <p className="mt-4 text-sm text-muted-foreground">Select two genomes from the catalog to compare.</p>}
          </div>
        </div>
      )}

      {detailGenome && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-slate-950/35 p-4">
          <div className="max-h-[85vh] w-full max-w-2xl overflow-auto rounded-md border border-border bg-card p-5 shadow-xl">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Genome Detail</h2>
              <button className="rounded-md border border-border px-3 py-2 text-sm" onClick={() => setDetailGenome(null)} type="button">Close</button>
            </div>
            <pre className="mt-4 max-h-[65vh] overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-50">
              {JSON.stringify(genomeDetail.data || detailGenome, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
