import type { ReactNode } from 'react'

type AppShellProps = {
  query: ReactNode
  map: ReactNode
  inspector: ReactNode
}

export const AppShell = ({ query, map, inspector }: AppShellProps) => {
  return (
    <div className="flex h-svh w-full flex-col overflow-hidden bg-canvas text-ink">
      <header className="flex shrink-0 flex-col gap-3 border-b border-line px-4 py-3 lg:flex-row lg:items-end lg:gap-8">
        <div className="flex items-baseline justify-between gap-4 lg:block lg:min-w-44">
          <h1 className="text-[15px] font-medium tracking-tight">
            Traffic Prediction
          </h1>
          <p className="text-[11px] uppercase tracking-[0.16em] text-muted">
            Mock data
          </p>
        </div>
        <div className="min-w-0 flex-1">{query}</div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section className="relative min-h-[50svh] flex-1 lg:min-h-0">{map}</section>
        <aside className="shrink-0 overflow-y-auto border-t border-line lg:w-[380px] lg:border-t-0 lg:border-l">
          {inspector}
        </aside>
      </div>
    </div>
  )
}
