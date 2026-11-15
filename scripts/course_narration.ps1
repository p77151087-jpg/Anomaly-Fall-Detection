param([string]$Storyboard = "outputs/course-film/storyboard.json")
$ErrorActionPreference = "Stop"
$storyPath = (Resolve-Path -LiteralPath $Storyboard).Path
$story = Get-Content -LiteralPath $storyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$outputRoot = Split-Path -Parent $storyPath
$voice = New-Object -ComObject SAPI.SpVoice
$chosen = $null
foreach ($token in $voice.GetVoices()) {
    if ($token.GetDescription() -eq $story.voice) { $chosen = $token }
}
if ($null -eq $chosen) { throw "Required installed narration voice is unavailable" }
$voice.Voice = $chosen
$voice.Rate = 1
foreach ($scene in $story.scenes) {
    $audioPath = Join-Path $outputRoot $scene.audio
    if (Test-Path -LiteralPath $audioPath) { throw "Narration already exists: $audioPath" }
    $stream = New-Object -ComObject SAPI.SpFileStream
    try {
        $stream.Open($audioPath, 3, $false)
        $voice.AudioOutputStream = $stream
        $null = $voice.Speak($scene.narration, 0)
    } finally {
        $stream.Close()
    }
    Write-Output $audioPath
}
