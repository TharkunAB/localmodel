type SearchResult = { url: string; title: string; content: string; time: Record<string, never> }
type SearchProvider = {
  id: string
  name: string
  execute(input: { query: string }, options: { signal: AbortSignal }): Promise<SearchResult[]>
}
type PluginContext = {
  websearch: { transform(edit: (editor: { add(provider: SearchProvider): void }) => void): Promise<unknown> }
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

export default {
  id: "localmodel.websearch.searxng",
  async setup(ctx: PluginContext): Promise<void> {
    await ctx.websearch.transform((editor) => editor.add({
      id: "searxng",
      name: "Local SearXNG",
      async execute({ query }, { signal }) {
        const response = await fetch("http://127.0.0.1:8888/search", {
          method: "POST",
          body: new URLSearchParams({ q: query, format: "json" }),
          headers: { Accept: "application/json" },
          redirect: "error",
          signal: AbortSignal.any([signal, AbortSignal.timeout(25_000)]),
        })
        if (!response.ok) throw new Error(`Local SearXNG returned HTTP ${response.status}`)
        const data: unknown = await response.json()
        if (!record(data) || !Array.isArray(data.results)) throw new Error("Invalid local SearXNG response")
        if (Array.isArray(data.unresponsive_engines) && data.unresponsive_engines.length > 0) {
          throw new Error("A configured search engine failed in local SearXNG; no fallback was used")
        }
        return data.results.flatMap((value: unknown): SearchResult[] => {
          if (!record(value) || typeof value.url !== "string") return []
          let url: URL
          try { url = new URL(value.url) } catch { return [] }
          if (url.protocol !== "https:" && url.protocol !== "http:") return []
          return [{
            url: url.href,
            title: typeof value.title === "string" ? value.title.slice(0, 300) : "",
            content: typeof value.content === "string" ? value.content.slice(0, 2_000) : "",
            time: {},
          }]
        }).slice(0, 8)
      },
    }))
  },
}
