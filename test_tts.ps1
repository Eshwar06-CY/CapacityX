
Add-Type -AssemblyName System.Speech
 = New-Object System.Speech.Synthesis.SpeechSynthesizer
.SetOutputToWaveFile('D:\Antigravity_Projects\CapacityX\test_tts.wav')
.Speak('CapacityX autonomous logistics agent pairing Strands Decider with Bedrock')
.Dispose()
