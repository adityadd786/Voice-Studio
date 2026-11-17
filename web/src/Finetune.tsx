import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Loader2, Mic, Square, Trash2, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { api, type Dataset, type ModelInfo } from '@/lib/api'
import { PROMPTS } from '@/lib/prompts'

const TARGET_MIN = 10 // minimum useful amount of speech, in minutes

export default function Finetune({ onModelsChanged }: { onModelsChanged: () => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [ds, setDs] = useState('')
  const [newName, setNewName] = useState('my-voice')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [modelName, setModelName] = useState('my-finetune')
  const [uploading, setUploading] = useState(false)
  const [building, setBuilding] = useState<{ stage: string; done: number; total: number } | null>(null)
  const [importing, setImporting] = useState(false)
  const [rec, setRec] = useState({ on: false, secs: 0 })
  const [prompt, setPrompt] = useState(0)
  const audioRef = useRef<HTMLInputElement>(null)
  const zipRef = useRef<HTMLInputElement>(null)
  const mediaRef = useRef<MediaRecorder | null>(null)
  const timerRef = useRef<number | undefined>(undefined)

  const refresh = useCallback(async () => {
    const [d, m] = await Promise.all([api.datasets(), api.models()])
    setDatasets(d)
    setModels(m)
    setDs((cur) => (d.some((x) => x.name === cur) ? cur : (d[0]?.name ?? '')))
  }, [])

  useEffect(() => { refresh().catch((e) => toast.error((e as Error).message)) }, [refresh])
  useEffect(() => () => window.clearInterval(timerRef.current), [])

  const current = datasets.find((d) => d.name === ds)
  const speechMin = current?.minutes ? current.minutes : current?.raw_minutes ?? 0
  const pct = Math.min(100, Math.round((speechMin / TARGET_MIN) * 100))

  async function onCreate() {
    const name = newName.trim().replace(/[^A-Za-z0-9_-]/g, '-')
    if (!name) return
    try {
      await api.createDataset(name)
      await refresh()
      setDs(name)
      toast.success(`Dataset "${name}" created`)
    } catch (e) { toast.error((e as Error).message) }
  }

  async function uploadAudio(blob: Blob, filename: string) {
    setUploading(true)
    try {
      await api.addDatasetAudio(ds, blob, filename)
      await refresh()
    } catch (e) { toast.error((e as Error).message) }
    finally { setUploading(false) }
  }

  async function onFiles(files: FileList | null) {
    if (!files || !ds) return
    for (const f of Array.from(files)) await uploadAudio(f, f.name)
    toast.success(`Added ${files.length} file(s)`)
    if (audioRef.current) audioRef.current.value = ''
  }

  async function startRec() {
    if (!ds) return toast.error('Create or choose a dataset first')
    try {
      // Turn off the browser's voice "enhancements": they change the sound of your voice.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
      const mr = new MediaRecorder(stream)
      const chunks: Blob[] = []
      mr.ondataavailable = (e) => chunks.push(e.data)
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        await uploadAudio(new Blob(chunks, { type: mr.mimeType }), 'recording.webm')
        toast.success('Recording saved to the dataset')
      }
      mr.start()
      mediaRef.current = mr
      setRec({ on: true, secs: 0 })
      timerRef.current = window.setInterval(() => setRec((r) => ({ ...r, secs: r.secs + 1 })), 1000)
    } catch {
      toast.error('Could not use the microphone. Allow microphone access in the browser.')
    }
  }

  function stopRec() {
    window.clearInterval(timerRef.current)
    mediaRef.current?.stop()
    setRec((r) => ({ ...r, on: false }))
  }

  async function onBuild() {
    if (!ds) return
    setBuilding({ stage: 'Starting…', done: 0, total: 0 })
    try {
      const { job } = await api.buildDataset(ds)
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500))
        const j = await api.job(job)
        setBuilding({ stage: j.stage ?? '', done: j.done, total: j.total })
        if (j.status === 'done') {
          toast.success(`Dataset ready: ${j.result?.segments} clips, ${j.result?.minutes} min`)
          break
        }
        if (j.status === 'error') throw new Error(j.error ?? 'Build failed')
        if (j.status === 'cancelled') break
      }
      await refresh()
    } catch (e) { toast.error((e as Error).message) }
    finally { setBuilding(null) }
  }

  async function onDeleteDataset() {
    if (!ds || !window.confirm(`Delete dataset "${ds}" and its audio?`)) return
    await api.deleteDataset(ds)
    await refresh()
  }

  async function onImport(file: File | undefined) {
    if (!file) return
    setImporting(true)
    try {
      const r = await api.importModel(modelName.trim() || 'my-finetune', file)
      toast.success(`Imported "${r.name}". Choose it in Studio under Voice & expression.`)
      await refresh()
      onModelsChanged()
    } catch (e) { toast.error((e as Error).message) }
    finally {
      setImporting(false)
      if (zipRef.current) zipRef.current.value = ''
    }
  }

  async function onDeleteModel(name: string) {
    if (!window.confirm(`Delete model "${name}"?`)) return
    await api.deleteModel(name)
    await refresh()
    onModelsChanged()
  }

  const mmss = `${Math.floor(rec.secs / 60)}:${String(rec.secs % 60).padStart(2, '0')}`

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
      <Card>
        <CardHeader>
          <CardTitle>Fine-tune a model on your voice</CardTitle>
          <CardDescription>
            A fine-tuned model learns your accent and speaking style, which a single reference clip cannot. Training needs a
            16 GB GPU, so it runs in free Google Colab. This page prepares the data and imports the result.
            Only use your own voice, or a voice you have permission to use.
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>1. Collect your speech</CardTitle>
          <CardDescription>
            Aim for <b>10–30 minutes</b> of clean, continuous speech in your natural accent. Less than 10 minutes tends to memorise the
            recording instead of learning your voice. Record in a quiet room, with no noise gate, no noise cancelling and no music.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-2">
            <div className="space-y-1">
              <Label>Dataset</Label>
              <select className="h-9 min-w-48 rounded-md border bg-background px-2 text-sm" value={ds}
                onChange={(e) => setDs(e.target.value)} aria-label="Dataset">
                {datasets.length === 0 && <option value="">No datasets yet</option>}
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
            </div>
            <Input className="w-44" value={newName} onChange={(e) => setNewName(e.target.value)} aria-label="New dataset name" />
            <Button variant="outline" onClick={onCreate}>New dataset</Button>
            {ds && <Button variant="ghost" size="icon" aria-label="Delete dataset" onClick={onDeleteDataset}><Trash2 className="size-4" /></Button>}
          </div>

          {ds && (
            <>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>{speechMin} min of speech</span>
                  <span className="text-muted-foreground">goal: {TARGET_MIN}+ min</span>
                </div>
                <Progress value={pct} />
              </div>

              <div className="flex flex-wrap gap-2">
                <input ref={audioRef} type="file" accept="audio/*,video/*" multiple hidden onChange={(e) => onFiles(e.target.files)} />
                <Button variant="outline" disabled={uploading} onClick={() => audioRef.current?.click()}>
                  {uploading ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />} Upload audio files
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Upload any recordings of you speaking (a lesson, a call, voice notes). Long files are cut into clips automatically.
              </p>

              <div className="space-y-3 rounded-lg border p-3">
                <div className="flex items-center justify-between">
                  <Label>Or record here: read this aloud</Label>
                  <Badge variant="secondary">{prompt + 1} / {PROMPTS.length}</Badge>
                </div>
                <p className="text-lg leading-relaxed">{PROMPTS[prompt]}</p>
                <div className="flex flex-wrap items-center gap-2">
                  {rec.on ? (
                    <Button variant="destructive" onClick={stopRec}><Square className="size-4" /> Stop and save ({mmss})</Button>
                  ) : (
                    <Button onClick={startRec} disabled={uploading}><Mic className="size-4" /> Start recording</Button>
                  )}
                  <Button variant="outline" onClick={() => setPrompt((p) => (p + 1) % PROMPTS.length)}>Next sentence</Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Keep recording and read several sentences in one go, with natural pauses. Save when you want a break.
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2. Build the dataset</CardTitle>
          <CardDescription>Cuts your audio into 3–11 second clips and writes out what you said using Whisper. This runs on your PC and can take several minutes for long recordings.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button onClick={onBuild} disabled={!ds || !!building || (current?.raw_minutes ?? 0) === 0}>
            {building ? <><Loader2 className="size-4 animate-spin" /> {building.stage}</> : 'Build dataset'}
          </Button>
          {building && building.total > 0 && <Progress value={Math.round((building.done / building.total) * 100)} />}
          {current?.built && !building && (
            <p className="text-sm">
              Ready: <b>{current.segments}</b> clips, <b>{current.minutes}</b> minutes of speech.
              {current.minutes < TARGET_MIN && (
                <span className="text-muted-foreground"> This is below {TARGET_MIN} minutes, so training would likely memorise it. Add more speech and build again.</span>
              )}
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>3. Train for free in Google Colab</CardTitle>
          <CardDescription>Needs a Google account. Training usually takes 20–60 minutes on the free T4 GPU.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <ol className="list-decimal space-y-1 pl-5 text-sm">
            <li>Download the dataset zip and the notebook below.</li>
            <li>Open <b>colab.research.google.com</b>, choose <b>File → Upload notebook</b>, and pick the notebook.</li>
            <li>Choose <b>Runtime → Change runtime type → T4 GPU</b>, then run the cells in order. Upload the dataset zip when asked.</li>
            <li>When it finishes, download <code>finetuned_xtts.zip</code> (about 1.9 GB) and import it in step 4.</li>
          </ol>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" disabled={!current?.built}
              onClick={() => { window.location.href = api.datasetZipUrl(ds) }}>
              <Download className="size-4" /> Dataset zip
            </Button>
            <Button variant="outline" onClick={() => { window.location.href = api.notebookUrl() }}>
              <Download className="size-4" /> Colab notebook
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Your recordings go to Google only if you upload the dataset zip to Colab. The notebook has not been run on Colab by the author of this app, so if a cell fails, copy the error message and ask for help.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>4. Import your trained model</CardTitle>
          <CardDescription>After importing, pick it in Studio → Settings → Voice &amp; expression → Model.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-2">
            <div className="space-y-1">
              <Label>Model name</Label>
              <Input className="w-48" value={modelName} onChange={(e) => setModelName(e.target.value)} />
            </div>
            <input ref={zipRef} type="file" accept=".zip" hidden onChange={(e) => onImport(e.target.files?.[0])} />
            <Button variant="outline" disabled={importing} onClick={() => zipRef.current?.click()}>
              {importing ? <><Loader2 className="size-4 animate-spin" /> Importing…</> : <><Upload className="size-4" /> Choose finetuned_xtts.zip</>}
            </Button>
          </div>
          {models.length === 0 && <p className="text-sm text-muted-foreground">No fine-tuned models yet.</p>}
          {models.map((m) => (
            <div key={m.name} className="flex items-center justify-between rounded-lg border p-3 text-sm">
              <span>{m.name} <span className="text-muted-foreground">· {m.size_mb} MB</span></span>
              <Button variant="ghost" size="icon" aria-label={`Delete ${m.name}`} onClick={() => onDeleteModel(m.name)}>
                <Trash2 className="size-4" />
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
