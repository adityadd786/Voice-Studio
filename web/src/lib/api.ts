export type Voice = { name: string; duration: number }
export type GenResult = { file: string; seconds: number; elapsed: number; rtf: number }
export type GenParams = {
  voice: string
  text: string
  lang: string
  engine: string
  speed: number
  gap: number
  duration: number | null
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  status: () => fetch('/api/status').then(json<{ cuda: boolean; gpu: string | null }>),
  voices: () => fetch('/api/voices').then(json<Voice[]>),
  addVoice: (name: string, file: File) => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', file)
    return fetch('/api/voices', { method: 'POST', body: fd }).then(json<Voice>)
  },
  deleteVoice: (name: string) =>
    fetch(`/api/voices/${encodeURIComponent(name)}`, { method: 'DELETE' }).then(json<{ ok: boolean }>),
  generate: (p: GenParams) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    }).then(json<GenResult>),
  audioUrl: (file: string) => `/api/audio/${encodeURIComponent(file)}`,
  voiceUrl: (name: string) => `/api/voices/${encodeURIComponent(name)}/audio`,
}
