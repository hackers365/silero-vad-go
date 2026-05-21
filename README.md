<h1 align="center">
  <br>
  silero-vad-go
  <br>
</h1>
<h4 align="center">A simple Golang (CGO + ONNX Runtime) speech detector powered by Silero VAD</h4>
<p align="center">
  <a href="https://pkg.go.dev/github.com/hackers365/silero-vad-go"><img src="https://pkg.go.dev/badge/github.com/hackers365/silero-vad-go.svg" alt="Go Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<br>

### Requirements

- [Golang](https://go.dev/doc/install) >= v1.21
- A C compiler (e.g. GCC)
- ONNX Runtime (v1.17 or newer)
- A [Silero VAD](https://github.com/snakers4/silero-vad) model (v5)

### Usage

For multi-stream applications, create one shared runtime and then create one stream per audio source.
The runtime owns the ONNX Runtime environment and a small pool of sessions, while each stream owns its
own VAD state.

```go
package main

import "github.com/hackers365/silero-vad-go/speech"

func main() {
	rt, err := speech.NewRuntime(speech.RuntimeConfig{
		ModelPath:   "testfiles/silero_vad.onnx",
		NumSessions: 2,
	})
	if err != nil {
		panic(err)
	}
	defer rt.Destroy()

	stream, err := rt.NewStream(speech.StreamConfig{
		SampleRate: 16000,
		Threshold:  0.5,
	})
	if err != nil {
		panic(err)
	}

	var pcm []float32 // Fill with 16 kHz PCM samples.
	segments, err := stream.Detect(pcm)
	if err != nil {
		panic(err)
	}

	_ = segments
}
```

`NumSessions` controls how many model sessions are loaded and shared across streams. The default is `1`.
Each `Stream` should be used serially by a single audio source. Different streams can run concurrently
and share the same runtime.

The legacy `Detector` API is still available for simple single-stream use:

```go
detector, err := speech.NewDetector(speech.DetectorConfig{
	ModelPath:  "testfiles/silero_vad.onnx",
	SampleRate: 16000,
	Threshold:  0.5,
})
if err != nil {
	panic(err)
}
defer detector.Destroy()

var pcm []float32 // Fill with 16 kHz PCM samples.
segments, err := detector.Detect(pcm)
```

### Development

In order to build and/or run this library, you need to export (or pass) some env variables to point to the ONNX runtime files.

#### Linux

```sh
LD_RUN_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
```

#### Darwin (MacOS)

```sh
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
sudo update_dyld_shared_cache
```

#### Windows

Install a GCC toolchain such as MSYS2 UCRT64 or MinGW-w64, and make sure `gcc.exe` is available in `PATH`.
Then point CGO to the ONNX Runtime headers and import library:

```powershell
$env:CGO_ENABLED = "1"
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
$env:C_INCLUDE_PATH = "E:\onnxruntime-win-x64-1.21.0\include"
$env:LIBRARY_PATH = "E:\onnxruntime-win-x64-1.21.0\lib"
```

Windows loads DLLs from the executable directory before `PATH`, but may load an older `onnxruntime.dll`
from `C:\Windows\System32` before checking `PATH`. For applications, copy the matching
`onnxruntime.dll` next to your built `.exe`. For tests on a machine with an older system DLL, put
`onnxruntime.dll` in the repository root, build the test binary into the root, and run it from the
package directory:

```powershell
go test -c -o .\silero-vad-go-speech.test.exe ./speech
Push-Location .\speech
..\silero-vad-go-speech.test.exe "-test.v" "-test.failfast"
Pop-Location
```

### License

MIT License - see [LICENSE](LICENSE) for full text

