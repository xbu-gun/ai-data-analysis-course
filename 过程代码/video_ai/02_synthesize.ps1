$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$videoConfig = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'storyboard.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$narrationDir = Join-Path $PSScriptRoot 'work\narration'
New-Item -ItemType Directory -Force -Path $narrationDir | Out-Null
$narrator = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $narrator.SelectVoice($videoConfig.voice)
    $narrator.Rate = $videoConfig.voice_rate
    $narrator.Volume = 100
    foreach ($scene in $videoConfig.scenes) {
        foreach ($cue in $scene.cues) {
            $wavePath = Join-Path $narrationDir ($cue.id + '.wav')
            $narrator.SetOutputToWaveFile($wavePath)
            $narrator.Speak($cue.speech)
            $narrator.SetOutputToNull()
            if ((Get-Item -LiteralPath $wavePath).Length -lt 1000) { throw "旁白生成失败：$($cue.id)" }
            Write-Output "旁白段 $($cue.id) 已生成"
        }
    }
} finally {
    $narrator.Dispose()
}
