import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AudioLines, Check, Download, FileText, Loader2, Mic, Pencil, RotateCcw, Square, Trash2, Upload, X } from 'lucide-react'
import { toast } from 'sonner'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Toaster } from '@/components/ui/sonner'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { api, type Settings, type Take, type Voice } from '@/lib/api'
import { DEFAULTS, LANGS, MARKERS, PHRASES, PRESETS, SLANG_EXAMPLES, TEMPLATES } from '@/lib/presets'

type HistoryItem = { id: string; voice: string; text: string; elapsed: number; takes: Take[] }

const SAMPLE =
  "Hello, this is my cloned voice. [pause 0.6s] [speed 0.85] This part is a little slower. [speed 1.15] And this part is faster.\n\nThis is a new paragraph, so there's a longer pause before it."

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? { ...fallback, ...JSON.parse(v) } : fallback
  } catch {
    return fallback
  }
}

function SliderField(p: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; fmt?: (v: number) => string; hint?: string; disabled?: boolean
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{p.label}</Label>
        <span className="text-sm tabular-nums text-muted-foreground">{p.fmt ? p.fmt(p.value) : p.value}</span>
      </div>
      <Slider min={p.min} max={p.max} step={p.step} value={[p.value]} disabled={p.disabled}
        onValueChange={(v) => p.onChange(Array.isArray(v) ? v[0] : v)} />
      {p.hint && <p className="text-xs text-muted-foreground">{p.hint}</p>}
    </div>
  )
}

function SwitchField(p: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="space-y-0.5">
        <Label>{p.label}</Label>
        {p.hint && <p className="text-xs text-muted-foreground">{p.hint}</p>}
      </div>
      <Switch checked={p.checked} onCheckedChange={p.onChange} />
    </div>
  )
}

export default function App() {
  const [gpu, setGpu] = useState<string | null | undefined>(undefined)
  const [voices, setVoices] = useState<Voice[]>([])
  const [voice, setVoice] = useState('')
  const [text, setText] = useState(() => localStorage.getItem('vs.text') ?? SAMPLE)
  const [s, setS] = useState<Settings>(() => load('vs.settings', DEFAULTS))
  const [preset, setPreset] = useState('Default')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0, status: '' })
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [pending, setPending] = useState<File | null>(null)   // chosen but not yet saved
  const [pendingName, setPendingName] = useState('')
  const [editing, setEditing] = useState<string | null>(null)  // voice being renamed
  const [editName, setEditName] = useState('')
  const jobRef = useRef<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const txtRef = useRef<HTMLInputElement>(null)
  const areaRef = useRef<HTMLTextAreaElement>(null)

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setS((o) => ({ ...o, [k]: v }))

  useEffect(() => { try { localStorage.setItem('vs.settings', JSON.stringify(s)) } catch { /* ignore */ } }, [s])
  useEffect(() => { try { localStorage.setItem('vs.text', text) } catch { /* ignore */ } }, [text])

  const loadVoices = useCallback(async () => {
    const v = await api.voices()
    setVoices(v)
    setVoice((cur) => (v.some((x) => x.name === cur) ? cur : (v[0]?.name ?? '')))
  }, [])

  useEffect(() => {
    document.documentElement.classList.add('dark')
    api.status().then((r) => setGpu(r.cuda ? r.gpu : null)).catch(() => setGpu(null))
    loadVoices().catch(() => toast.error('Backend not reachable. Start the API server on port 8000.'))
  }, [loadVoices])

  const stats = useMemo(() => {
    const clean = text.replace(/\[[^\]]*\]/g, ' ')
    const words = clean.trim() ? clean.trim().split(/\s+/).length : 0
    const secs = s.fit ? s.duration : (words / (155 * s.speed)) * 60
    return { words, chars: text.length, secs: Math.round(secs) }
  }, [text, s.speed, s.fit, s.duration])

  function applyPreset(name: string) {
    setPreset(name)
    setS((o) => ({ ...o, ...PRESETS[name] }))
  }

  function insertMarker(m: string) {
    const el = areaRef.current
    const start = el?.selectionStart ?? text.length
    const end = el?.selectionEnd ?? text.length
    const next = text.slice(0, start) + m + ' ' + text.slice(end)
    setText(next)
    requestAnimationFrame(() => {
      el?.focus()
      const pos = start + m.length + 1
      el?.setSelectionRange(pos, pos)
    })
  }

  function onPick(file: File | undefined) {
    if (!file) return
    setPending(file)
    setPendingName(file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim().slice(0, 40) || 'My voice')
    if (fileRef.current) fileRef.current.value = ''
  }

  async function onSaveUpload() {
    if (!pending || !pendingName.trim()) return
    try {
      const v = await api.addVoice(pendingName, pending)
      toast.success(`Saved voice "${v.name}"`)
      setPending(null)
      await loadVoices()
      setVoice(v.name)
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  async function onRename(oldName: string) {
    const target = editName.trim()
    if (!target || target === oldName) return setEditing(null)
    try {
      const v = await api.renameVoice(oldName, target)
      toast.success(`Renamed to "${v.name}". Update any [voice ${oldName}] markers in your script.`)
      setEditing(null)
      await loadVoices()
      if (voice === oldName) setVoice(v.name)
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  async function onDelete(name: string) {
    await api.deleteVoice(name)
    toast.success(`Removed "${name}"`)
    loadVoices()
  }

  async function onTxt(file: File | undefined) {
    if (!file) return
    setText(await file.text())
    if (txtRef.current) txtRef.current.value = ''
  }

  async function onGenerate() {
    if (!voice) return toast.error('Choose or upload a voice first')
    setBusy(true)
    setProgress({ done: 0, total: 0, status: 'queued' })
    try {
      const { job } = await api.generate(voice, text, s)
      jobRef.current = job
      for (;;) {
        await new Promise((r) => setTimeout(r, 1000))
        const j = await api.job(job)
        setProgress({ done: j.done, total: j.total, status: j.status })
        if (j.status === 'done' && j.result) {
          setHistory((h) => [{ id: job, voice, text, elapsed: j.result!.elapsed, takes: j.result!.takes }, ...h])
          toast.success(`Done in ${j.result.elapsed}s`)
          break
        }
        if (j.status === 'error') throw new Error(j.error ?? 'Generation failed')
        if (j.status === 'cancelled') { toast('Cancelled'); break }
      }
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      jobRef.current = null
      setBusy(false)
    }
  }

  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0
  const field = (label: string, child: ReactNode) => <div className="space-y-2"><Label>{label}</Label>{child}</div>

  return (
    <div className="min-h-screen">
      <Toaster />
      <header className="border-b">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2">
            <AudioLines className="size-5" />
            <h1 className="text-lg font-semibold tracking-tight">Voice Studio</h1>
          </div>
          <Badge variant={gpu ? 'default' : 'secondary'}>{gpu === undefined ? 'Checking…' : gpu ?? 'CPU only'}</Badge>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-[300px_1fr]">
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Voices</CardTitle>
            <CardDescription>10–30 s of clean speech from one person. Only clone voices you have consent to use.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {voices.length === 0 && <p className="text-sm text-muted-foreground">No voices yet.</p>}
            {voices.map((v) => (
              <div key={v.name} className={`rounded-lg border p-3 ${voice === v.name ? 'border-primary' : ''}`}>
                <div className="flex items-center justify-between gap-2">
                  {editing === v.name ? (
                    <>
                      <Input autoFocus value={editName} maxLength={40} aria-label="Voice name"
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') onRename(v.name); if (e.key === 'Escape') setEditing(null) }} />
                      <Button variant="ghost" size="icon" aria-label="Save name" onClick={() => onRename(v.name)}>
                        <Check className="size-4" />
                      </Button>
                      <Button variant="ghost" size="icon" aria-label="Cancel rename" onClick={() => setEditing(null)}>
                        <X className="size-4" />
                      </Button>
                    </>
                  ) : (
                    <>
                      <button className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm font-medium" onClick={() => setVoice(v.name)}>
                        <Mic className="size-4 shrink-0" /> <span className="truncate">{v.name}</span>
                        <span className="text-xs font-normal text-muted-foreground">{v.duration}s</span>
                      </button>
                      <Button variant="ghost" size="icon" aria-label={`Rename ${v.name}`}
                        onClick={() => { setEditing(v.name); setEditName(v.name) }}>
                        <Pencil className="size-4" />
                      </Button>
                      <Button variant="ghost" size="icon" aria-label={`Delete ${v.name}`} onClick={() => onDelete(v.name)}>
                        <Trash2 className="size-4" />
                      </Button>
                    </>
                  )}
                </div>
                <audio controls preload="none" src={api.voiceUrl(v.name)} className="mt-2 h-8 w-full" />
              </div>
            ))}
            <input ref={fileRef} type="file" accept="audio/*" hidden onChange={(e) => onPick(e.target.files?.[0])} />
            {pending ? (
              <div className="space-y-2 rounded-lg border border-dashed p-3">
                <Label>Name this voice</Label>
                <Input autoFocus value={pendingName} maxLength={40}
                  onChange={(e) => setPendingName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') onSaveUpload(); if (e.key === 'Escape') setPending(null) }} />
                <p className="truncate text-xs text-muted-foreground">File: {pending.name}</p>
                <div className="flex gap-2">
                  <Button className="flex-1" disabled={!pendingName.trim()} onClick={onSaveUpload}>Save voice</Button>
                  <Button variant="outline" onClick={() => setPending(null)}>Cancel</Button>
                </div>
              </div>
            ) : (
              <Button variant="outline" className="w-full" onClick={() => fileRef.current?.click()}>
                <Upload className="size-4" /> Upload voice
              </Button>
            )}
            <p className="text-xs text-muted-foreground">
              Use <code>[voice name]</code> in the script to switch between voices mid-text.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Script</CardTitle>
              <CardDescription>Click a marker to insert it at the cursor. Blank lines add a paragraph pause.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {MARKERS.map((m) => (
                  <Button key={m.label} variant="outline" size="sm" onClick={() => insertMarker(m.text)} title={m.text}>
                    {m.label}
                  </Button>
                ))}
                <input ref={txtRef} type="file" accept=".txt,text/plain" hidden onChange={(e) => onTxt(e.target.files?.[0])} />
                <Button variant="ghost" size="sm" onClick={() => txtRef.current?.click()}>
                  <FileText className="size-4" /> Load .txt
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-muted-foreground">Start from a template:</span>
                {TEMPLATES.map((t) => (
                  <Button key={t.label} variant="secondary" size="sm" onClick={() => setText(t.text)}>{t.label}</Button>
                ))}
              </div>
              <details className="rounded-lg border px-3 py-2">
                <summary className="cursor-pointer text-sm">Friendly phrases (click to insert)</summary>
                <div className="mt-2 flex flex-wrap gap-2">
                  {PHRASES.map((p) => (
                    <Button key={p.label} variant="outline" size="sm" onClick={() => insertMarker(p.text)}>{p.label}</Button>
                  ))}
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Hindi phrases are written in Devanagari with <code>[lang hi]</code> … <code>[lang en]</code> so they are pronounced properly.
                  Hindi in Roman letters would be read with English sounds.
                </p>
              </details>
              <Textarea ref={areaRef} rows={9} value={text} onChange={(e) => setText(e.target.value)}
                placeholder="Type what the voice should say…" />
              <p className="text-xs text-muted-foreground">
                {stats.words} words · {stats.chars} characters · about {Math.floor(stats.secs / 60)}:{String(stats.secs % 60).padStart(2, '0')} of audio
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-4">
              <div>
                <CardTitle>Settings</CardTitle>
                <CardDescription>Saved automatically in this browser.</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Select value={preset} onValueChange={(v) => v && applyPreset(v)}>
                  <SelectTrigger className="w-44"><SelectValue>{preset}</SelectValue></SelectTrigger>
                  <SelectContent>
                    {Object.keys(PRESETS).map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}
                  </SelectContent>
                </Select>
                <Button variant="ghost" size="icon" aria-label="Reset settings" onClick={() => { setS(DEFAULTS); setPreset('Default') }}>
                  <RotateCcw className="size-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <Accordion multiple defaultValue={['pace']}>
                <AccordionItem value="pace">
                  <AccordionTrigger>Pace & timing</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    <SliderField label="Speed" value={s.speed} min={0.5} max={2} step={0.05} disabled={s.fit}
                      fmt={(v) => `${v.toFixed(2)}x`} onChange={(v) => set('speed', v)}
                      hint="Around 0.8–1.3x sounds natural. Override any part with [speed 0.9]." />
                    <SliderField label="Pause between sentences" value={s.gap} min={0} max={2} step={0.05}
                      fmt={(v) => `${v.toFixed(2)}s`} onChange={(v) => set('gap', v)} />
                    <SliderField label="Pause between paragraphs" value={s.paraGap} min={0} max={4} step={0.1}
                      fmt={(v) => `${v.toFixed(1)}s`} onChange={(v) => set('paraGap', v)} />
                    <SwitchField label="Fit to exact duration" hint="Stretches the result (pitch preserved) to a target length, e.g. for video dubbing."
                      checked={s.fit} onChange={(v) => set('fit', v)} />
                    {s.fit && (
                      <SliderField label="Target duration" value={s.duration} min={2} max={600} step={1}
                        fmt={(v) => `${v}s`} onChange={(v) => set('duration', v)} />
                    )}
                    <SwitchField label="Trim silence" hint="Removes dead air at the start and end of each sentence."
                      checked={s.trim} onChange={(v) => set('trim', v)} />
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="style">
                  <AccordionTrigger>Course & explanation style</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    <SwitchField label="Teaching-style pacing" checked={s.smartPace} onChange={(v) => set('smartPace', v)}
                      hint="Pauses after questions and colons, a longer pause at “…”, and slightly slower delivery for definitions and key points (“for example”, “remember”, “step 2”)." />
                    <SwitchField label="Tech & course terms" checked={s.techTerms} onChange={(v) => set('techTerms', v)}
                      hint="Reads API, SQL, AWS, UPI, GST, GitHub, Kubernetes and similar terms correctly. Your own dictionary entries win." />
                    <SwitchField label="Indian amounts (lakh / crore)" checked={s.indianText} onChange={(v) => set('indianText', v)}
                      hint="₹5,00,000 is read as “five lakh rupees” and Rs. 2.5 crore as “two point five crore rupees”." />
                    <SwitchField label="Casual speech (English)" checked={s.casual} onChange={(v) => set('casual', v)}
                      hint="“going to” becomes “gonna”, “want to” becomes “wanna”, and so on." />
                    <p className="text-xs text-muted-foreground">
                      Tip: write for the ear. Short sentences, one idea each, read much better than long ones. Listen once, then fix the places that sound rushed with a <code>[pause]</code> or <code>[speed 0.9]</code>.
                    </p>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="voice">
                  <AccordionTrigger>Voice & expression</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    <SliderField label="Expressiveness (temperature)" value={s.temperature} min={0.1} max={1.2} step={0.05}
                      fmt={(v) => v.toFixed(2)} onChange={(v) => set('temperature', v)}
                      hint="Low = steady and consistent. High = more emotion and variation, but more glitches." />
                    <SliderField label="Pitch" value={s.pitch} min={-6} max={6} step={0.5}
                      fmt={(v) => `${v > 0 ? '+' : ''}${v} st`} onChange={(v) => set('pitch', v)}
                      hint="Semitones. Moves the whole voice up or down; override with [pitch +2]." />
                    <SliderField label="Reference analysis length" value={s.refSeconds} min={6} max={30} step={1}
                      fmt={(v) => `${v}s`} onChange={(v) => set('refSeconds', v)}
                      hint="How much of your reference clip is studied. Longer usually clones more closely." />
                    <p className="text-xs text-muted-foreground">
                      Accent and speaking style are copied from your reference clip, not from any setting. For an Indian-English accent, use a clean clip of an Indian-English speaker (your own voice works). For a friendlier, more casual delivery, record the clip in that tone.
                    </p>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="lang">
                  <AccordionTrigger>Language & slang</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    {field('Language', (
                      <Select value={s.lang} onValueChange={(v) => v && set('lang', v)}>
                        <SelectTrigger className="w-full"><SelectValue>{LANGS[s.lang]}</SelectValue></SelectTrigger>
                        <SelectContent>
                          {Object.entries(LANGS).map(([k, n]) => <SelectItem key={k} value={k}>{n}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    ))}
                    <p className="text-xs text-muted-foreground">
                      Switch language mid-script with <code>[lang hi]</code>. Write each language in its own script (e.g. Hindi in Devanagari).
                    </p>
                    <div className="space-y-2">
                      <Label>Pronunciation dictionary</Label>
                      <Textarea rows={4} value={s.pronunciations} onChange={(e) => set('pronunciations', e.target.value)}
                        placeholder={SLANG_EXAMPLES} />
                      <p className="text-xs text-muted-foreground">
                        One per line: <code>word = how to say it</code>. Use it for slang, brand names and acronyms. Spell the sound out until it sounds right.
                      </p>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="output">
                  <AccordionTrigger>Output</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    <SwitchField label="Normalize loudness" hint="Even volume, about −16 LUFS (typical for video and podcasts)."
                      checked={s.normalize} onChange={(v) => set('normalize', v)} />
                    <SliderField label="Volume" value={s.gainDb} min={-12} max={12} step={1}
                      fmt={(v) => `${v > 0 ? '+' : ''}${v} dB`} onChange={(v) => set('gainDb', v)} />
                    <div className="grid gap-4 sm:grid-cols-2">
                      {field('Takes', (
                        <Select value={String(s.takes)} onValueChange={(v) => v && set('takes', Number(v))}>
                          <SelectTrigger className="w-full"><SelectValue>{s.takes} {s.takes === 1 ? 'take' : 'takes'}</SelectValue></SelectTrigger>
                          <SelectContent>
                            {[1, 2, 3, 4].map((n) => <SelectItem key={n} value={String(n)}>{n} {n === 1 ? 'take' : 'takes'}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      ))}
                      {field('Format', (
                        <Select value={s.format} onValueChange={(v) => v && set('format', v as Settings['format'])}>
                          <SelectTrigger className="w-full"><SelectValue>{s.format.toUpperCase()}</SelectValue></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="wav">WAV (lossless)</SelectItem>
                            <SelectItem value="mp3">MP3 (smaller)</SelectItem>
                          </SelectContent>
                        </Select>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground">Several takes give different readings of the same text so you can pick the best. Each take takes extra time.</p>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="advanced">
                  <AccordionTrigger>Advanced</AccordionTrigger>
                  <AccordionContent className="space-y-5 pt-2">
                    {field('Seed (blank = random)', (
                      <Input inputMode="numeric" value={s.seed} placeholder="e.g. 42"
                        onChange={(e) => set('seed', e.target.value.replace(/[^0-9]/g, ''))} />
                    ))}
                    <p className="text-xs text-muted-foreground">Same seed and settings give the same reading. Each result shows its seed so you can reuse a good one.</p>
                    <SliderField label="Top-p" value={s.topP} min={0.1} max={1} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => set('topP', v)} />
                    <SliderField label="Top-k" value={s.topK} min={1} max={100} step={1} onChange={(v) => set('topK', v)} />
                    <SliderField label="Repetition penalty" value={s.repetitionPenalty} min={1} max={20} step={0.5}
                      fmt={(v) => v.toFixed(1)} onChange={(v) => set('repetitionPenalty', v)}
                      hint="Raise it if the voice stutters or repeats words." />
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="space-y-3 pt-6">
              {busy && (
                <div className="space-y-2">
                  <Progress value={progress.total ? pct : null} />
                  <p className="text-xs text-muted-foreground">
                    {progress.status === 'queued' || !progress.total
                      ? 'Loading model and analysing the voice…'
                      : `Sentence ${progress.done} of ${progress.total} (${pct}%)`}
                  </p>
                </div>
              )}
              <div className="flex gap-2">
                <Button onClick={onGenerate} disabled={busy || !text.trim() || !voice} className="flex-1">
                  {busy ? <><Loader2 className="size-4 animate-spin" /> Generating…</> : 'Generate'}
                </Button>
                {busy && (
                  <Button variant="outline" onClick={() => jobRef.current && api.cancel(jobRef.current)}>
                    <Square className="size-4" /> Cancel
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {history.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Results</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {history.map((h) => (
                  <div key={h.id} className="space-y-3 rounded-lg border p-3">
                    <p className="line-clamp-2 text-sm">{h.text}</p>
                    {h.takes.map((t, i) => (
                      <div key={t.file} className="space-y-1">
                        <audio controls src={api.audioUrl(t.file)} className="w-full" />
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>
                            {h.takes.length > 1 ? `Take ${i + 1} · ` : ''}{h.voice} · {t.seconds}s · seed {t.seed}
                            {i === 0 ? ` · made in ${h.elapsed}s` : ''}
                          </span>
                          <a href={api.audioUrl(t.file)} download={t.file} className="inline-flex items-center gap-1 text-foreground underline">
                            <Download className="size-3" /> Download
                          </a>
                        </div>
                      </div>
                    ))}
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
