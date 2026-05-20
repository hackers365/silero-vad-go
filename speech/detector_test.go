package speech

import (
	"encoding/binary"
	"log/slog"
	"math"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func readSamplesFromFile(t *testing.T, path string) []float32 {
	t.Helper()

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	samples := make([]float32, 0, len(data)/4)
	for i := 0; i < len(data); i += 4 {
		samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
	}
	return samples
}

func TestDetectorConfigIsValid(t *testing.T) {
	tcs := []struct {
		name string
		cfg  DetectorConfig
		err  string
	}{
		{
			name: "missing ModelPath",
			cfg: DetectorConfig{
				ModelPath: "",
			},
			err: "invalid ModelPath: should not be empty",
		},
		{
			name: "invalid SampleRate",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 48000,
			},
			err: "invalid SampleRate: valid values are 8000 and 16000",
		},
		{
			name: "invalid Threshold",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0,
			},
			err: "invalid Threshold: should be in range (0, 1)",
		},
		{
			name: "invalid MinSilenceDurationMs",
			cfg: DetectorConfig{
				ModelPath:            "../testfiles/silero_vad.onnx",
				SampleRate:           16000,
				Threshold:            0.5,
				MinSilenceDurationMs: -1,
			},
			err: "invalid MinSilenceDurationMs: should be a positive number",
		},
		{
			name: "invalid SpeechPadMs",
			cfg: DetectorConfig{
				ModelPath:   "../testfiles/silero_vad.onnx",
				SampleRate:  16000,
				Threshold:   0.5,
				SpeechPadMs: -1,
			},
			err: "invalid SpeechPadMs: should be a positive number",
		},
		{
			name: "valid",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0.5,
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.IsValid()
			if tc.err != "" {
				require.EqualError(t, err, tc.err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestNewDetector(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)

	err = sd.Destroy()
	require.NoError(t, err)
}

func TestSpeechDetection(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		AddSource: true,
		Level:     slog.LevelDebug,
	}))
	slog.SetDefault(logger)

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)
	defer func() {
		require.NoError(t, sd.Destroy())
	}()

	samples := readSamplesFromFile(t, "../testfiles/samples.pcm")
	samples2 := readSamplesFromFile(t, "../testfiles/samples2.pcm")

	t.Run("detect", func(t *testing.T) {
		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.056,
				SpeechEndAt:   1.632,
			},
			{
				SpeechStartAt: 2.88,
				SpeechEndAt:   3.232,
			},
			{
				SpeechStartAt: 4.448,
				SpeechEndAt:   0,
			},
		}, segments)

		err = sd.Reset()
		require.NoError(t, err)

		segments, err = sd.Detect(samples2)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 3.008,
				SpeechEndAt:   6.24,
			},
			{
				SpeechStartAt: 7.072,
				SpeechEndAt:   8.16,
			},
		}, segments)
	})

	t.Run("reset", func(t *testing.T) {
		err = sd.Reset()
		require.NoError(t, err)

		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.056,
				SpeechEndAt:   1.632,
			},
			{
				SpeechStartAt: 2.88,
				SpeechEndAt:   3.232,
			},
			{
				SpeechStartAt: 4.448,
				SpeechEndAt:   0,
			},
		}, segments)
	})

	t.Run("speech padding", func(t *testing.T) {
		cfg.SpeechPadMs = 10
		sd, err := NewDetector(cfg)
		require.NoError(t, err)
		require.NotNil(t, sd)
		defer func() {
			require.NoError(t, sd.Destroy())
		}()

		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.056 - 0.01,
				SpeechEndAt:   1.632 + 0.01,
			},
			{
				SpeechStartAt: 2.88 - 0.01,
				SpeechEndAt:   3.232 + 0.01,
			},
			{
				SpeechStartAt: 4.448 - 0.01,
				SpeechEndAt:   0,
			},
		}, segments)
	})
}

func TestRuntimeSharedAcrossStreams(t *testing.T) {
	rt, err := NewRuntime(RuntimeConfig{
		ModelPath:         "../testfiles/silero_vad.onnx",
		NumSessions:       2,
		IntraOpNumThreads: 1,
		InterOpNumThreads: 1,
	})
	require.NoError(t, err)
	defer func() {
		require.NoError(t, rt.Destroy())
	}()

	cfg := StreamConfig{
		SampleRate: 16000,
		Threshold:  0.5,
	}
	stream1, err := rt.NewStream(cfg)
	require.NoError(t, err)
	stream2, err := rt.NewStream(cfg)
	require.NoError(t, err)

	samples := readSamplesFromFile(t, "../testfiles/samples.pcm")
	samples2 := readSamplesFromFile(t, "../testfiles/samples2.pcm")

	segments1, err := stream1.Detect(samples)
	require.NoError(t, err)
	require.Equal(t, []Segment{
		{
			SpeechStartAt: 1.056,
			SpeechEndAt:   1.632,
		},
		{
			SpeechStartAt: 2.88,
			SpeechEndAt:   3.232,
		},
		{
			SpeechStartAt: 4.448,
			SpeechEndAt:   0,
		},
	}, segments1)

	segments2, err := stream2.Detect(samples2)
	require.NoError(t, err)
	require.Equal(t, []Segment{
		{
			SpeechStartAt: 3.008,
			SpeechEndAt:   6.24,
		},
		{
			SpeechStartAt: 7.072,
			SpeechEndAt:   8.16,
		},
	}, segments2)
}

func TestRuntimeSharedAcrossConcurrentStreams(t *testing.T) {
	rt, err := NewRuntime(RuntimeConfig{
		ModelPath:         "../testfiles/silero_vad.onnx",
		NumSessions:       2,
		IntraOpNumThreads: 1,
		InterOpNumThreads: 1,
	})
	require.NoError(t, err)
	defer func() {
		require.NoError(t, rt.Destroy())
	}()

	cfg := StreamConfig{
		SampleRate: 16000,
		Threshold:  0.5,
	}
	stream1, err := rt.NewStream(cfg)
	require.NoError(t, err)
	stream2, err := rt.NewStream(cfg)
	require.NoError(t, err)

	samples := readSamplesFromFile(t, "../testfiles/samples.pcm")
	samples2 := readSamplesFromFile(t, "../testfiles/samples2.pcm")

	var wg sync.WaitGroup
	wg.Add(2)

	var segments1 []Segment
	var err1 error
	go func() {
		defer wg.Done()
		segments1, err1 = stream1.Detect(samples)
	}()

	var segments2 []Segment
	var err2 error
	go func() {
		defer wg.Done()
		segments2, err2 = stream2.Detect(samples2)
	}()

	wg.Wait()

	require.NoError(t, err1)
	require.NoError(t, err2)
	require.Equal(t, []Segment{
		{
			SpeechStartAt: 1.056,
			SpeechEndAt:   1.632,
		},
		{
			SpeechStartAt: 2.88,
			SpeechEndAt:   3.232,
		},
		{
			SpeechStartAt: 4.448,
			SpeechEndAt:   0,
		},
	}, segments1)
	require.Equal(t, []Segment{
		{
			SpeechStartAt: 3.008,
			SpeechEndAt:   6.24,
		},
		{
			SpeechStartAt: 7.072,
			SpeechEndAt:   8.16,
		},
	}, segments2)
}

func TestStreamDetectAfterRuntimeDestroy(t *testing.T) {
	rt, err := NewRuntime(RuntimeConfig{
		ModelPath: "../testfiles/silero_vad.onnx",
	})
	require.NoError(t, err)

	stream, err := rt.NewStream(StreamConfig{
		SampleRate: 16000,
		Threshold:  0.5,
	})
	require.NoError(t, err)
	require.NoError(t, rt.Destroy())

	samples := readSamplesFromFile(t, "../testfiles/samples.pcm")
	_, err = stream.Detect(samples)
	require.ErrorContains(t, err, "runtime is destroyed")
}
