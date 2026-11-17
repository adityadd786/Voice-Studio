import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioLines, Download, Loader2, Mic, Trash2, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Toaster } from '@/components/ui/sonner'
import { Textarea } from '@/components/ui/textarea'
import { api, type GenResult, type Voice } from '@/lib/api'

const LANGS: Record<string, string> = {
  en: 'English', hi: 'Hindi', es: 'Spanish', fr: 'French', de: 'German', it: 'Italian',
  pt: 'Portuguese', pl: 'Polish', tr: 'Turkish', ru: 'Russian', nl: 'Dutch', cs: 'Czech',
  ar: 'Arabic', 'zh-cn': 'Chinese', ja: 'Japanese', ko: 'Korean', hu: 'Hungarian',
}

type HistoryItem = GenResult & { voice: string; text: string }

const SAMPLE =
  'Hello, this is my cloned voice. [pause 0.6s] [speed 0.85] This part is a little slower. [speed 1.15] And this part is faster.'

export default function App() {
  const [gpu, setGpu] = useState<string | null | undefined>(undefined)
  const [voices, setVoices] = useState<Voice[]>([])
  const [voice, setVoice] = useState('')
  const [text, setText] = useState(SAMPLE)
  const [lang, setLang] = useState('en')
  const [speed, setSpeed] = useState(1)
  const [gap, setGap] = useState(0.25)
  const [fit, setFit] = useState(false)
  const [duration, setDuration] = useState(10)
  const [busy, setBusy] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const fileRef = useRef<HTMLInputElement>(null)

  const loadVoices = useCallback(async () => {
    const v = await api.voices()
    setVoices(v)
    setVoice((cur) => (v.some((x) => x.name === cur) ? cur : (v[0]?.name ?? '')))
  }, [])

  useEffect(() => {
    document.documentElement.classList.add('dark')
    api.status().then((s) => setGpu(s.cuda ? s.gpu : null)).catch(() => setGpu(null))
    loadVoices().catch(() => toast.error('Backend not reachable. Start the API server on port 8000.'))
  }, [loadVoices])

  async function onUpload(file: File | undefined) {
    if (!file) return
    const name = file.name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]/g, '_') || 'voice'
    try {
      await api.addVoice(name, file)
      toast.success(`Voice "${name}" added`)
      await loadVoices()
      setVoice(name)
    } catch (e) {
      toast.error((e as Error).message)
    }
    if (fileRef.current) fileRef.current.value = ''
  }

  async function onDelete(name: string) {
    await api.deleteVoice(name)
    toast.success(`Removed "${name}"`)
    loadVoices()
  }

  async function onGenerate() {
    if (!voice) return toast.error('Choose or upload a voice first')
    setBusy(true)
    try {
      const r = await api.generate({
        voice, text, lang, engine: 'xtts', speed, gap, duration: fit ? duration : null,
      })
      setHistory((h) => [{ ...r, voice, text }, ...h])
      toast.success(`Generated ${r.seconds}s in ${r.elapsed}s`)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Toaster />
      <header className="border-b">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2">
            <AudioLines className="size-5" />
            <h1 className="text-lg font-semibold tracking-tight">Voice Studio</h1>
          </div>
          <Badge variant={gpu ? 'default' : 'secondary'}>
            {gpu === undefined ? 'Checking…' : gpu ?? 'CPU only'}
          </Badge>
        </div>
      </header>

      <main className="mx-auto grid max-w-5xl gap-6 px-4 py-6 md:grid-cols-[320px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Voices</CardTitle>
            <CardDescription>Upload 10–30 s of clean speech. Only use voices you have consent to clone.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {voices.length === 0 && <p className="text-sm text-muted-foreground">No voices yet.</p>}
            {voices.map((v) => (
              <div
                key={v.name}
                className={`rounded-lg border p-3 ${voice === v.name ? 'border-primary' : ''}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <button className="flex items-center gap-2 text-left text-sm font-medium" onClick={() => setVoice(v.name)}>
                    <Mic className="size-4" /> {v.name}
                    <span className="text-xs font-normal text-muted-foreground">{v.duration}s</span>
                  </button>
                  <Button variant="ghost" size="icon" aria-label={`Delete ${v.name}`} onClick={() => onDelete(v.name)}>
                    <Trash2 className="size-4" />
                  </Button>
                </div>
                <audio controls preload="none" src={api.voiceUrl(v.name)} className="mt-2 h-8 w-full" />
              </div>
            ))}
            <input ref={fileRef} type="file" accept="audio/*" hidden onChange={(e) => onUpload(e.target.files?.[0])} />
            <Button variant="outline" className="w-full" onClick={() => fileRef.current?.click()}>
              <Upload className="size-4" /> Upload voice
            </Button>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Generate speech</CardTitle>
              <CardDescription>
                Markers: <code>[pause 0.5s]</code> adds silence, <code>[speed 0.9]</code> changes pace for the text after it.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <Textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} placeholder="Type what the voice should say…" />

              <div className="grid gap-5 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label>Language</Label>
                  <Select value={lang} onValueChange={(v) => v && setLang(v)}>
                    <SelectTrigger className="w-full"><SelectValue>{LANGS[lang]}</SelectValue></SelectTrigger>
                    <SelectContent>
                      {Object.entries(LANGS).map(([k, n]) => <SelectItem key={k} value={k}>{n}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Pause between sentences: {gap.toFixed(2)}s</Label>
                  <Slider min={0} max={2} step={0.05} value={[gap]} onValueChange={(v) => setGap(Array.isArray(v) ? v[0] : v)} />
                </div>
                <div className="space-y-2">
                  <Label>Speed: {speed.toFixed(2)}x</Label>
                  <Slider min={0.5} max={2} step={0.05} value={[speed]} disabled={fit}
                    onValueChange={(v) => setSpeed(Array.isArray(v) ? v[0] : v)} />
                </div>
                <div className="space-y-2">
                  <Label className="flex items-center gap-2">
                    <input type="checkbox" checked={fit} onChange={(e) => setFit(e.target.checked)} />
                    Fit to duration: {duration}s
                  </Label>
                  <Slider min={2} max={120} step={1} value={[duration]} disabled={!fit}
                    onValueChange={(v) => setDuration(Array.isArray(v) ? v[0] : v)} />
                </div>
              </div>

              <Button onClick={onGenerate} disabled={busy || !text.trim()} className="w-full">
                {busy ? <><Loader2 className="size-4 animate-spin" /> Generating… (first run loads the model)</> : 'Generate'}
              </Button>
            </CardContent>
          </Card>

          {history.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Results</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {history.map((h) => (
                  <div key={h.file} className="space-y-2 rounded-lg border p-3">
                    <p className="line-clamp-2 text-sm">{h.text}</p>
                    <audio controls src={api.audioUrl(h.file)} className="w-full" />
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{h.voice} · {h.seconds}s audio · made in {h.elapsed}s ({h.rtf}x real time)</span>
                      <a href={api.audioUrl(h.file)} download={h.file} className="inline-flex items-center gap-1 text-foreground underline">
                        <Download className="size-3" /> Download
                      </a>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
