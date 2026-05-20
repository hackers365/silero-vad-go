package speech

import "fmt"

// DetectorConfig keeps the legacy one-detector-per-stream constructor working.
type DetectorConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence to wait for each speech segment before separating it.
	MinSilenceDurationMs int
	// The padding to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
	// The loglevel for the onnx environment, by default it is set to LogLevelWarn.
	LogLevel LogLevel
}

func (c DetectorConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if err := c.streamConfig().IsValid(); err != nil {
		return err
	}

	return nil
}

func (c DetectorConfig) runtimeConfig() RuntimeConfig {
	return RuntimeConfig{
		ModelPath:         c.ModelPath,
		LogLevel:          c.LogLevel,
		NumSessions:       1,
		IntraOpNumThreads: 1,
		InterOpNumThreads: 1,
	}
}

func (c DetectorConfig) streamConfig() StreamConfig {
	return StreamConfig{
		SampleRate:           c.SampleRate,
		Threshold:            c.Threshold,
		MinSilenceDurationMs: c.MinSilenceDurationMs,
		SpeechPadMs:          c.SpeechPadMs,
	}
}

// Detector is the legacy wrapper around a shared Runtime and one Stream.
type Detector struct {
	runtime *Runtime
	stream  *Stream
}

func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	rt, err := NewRuntime(cfg.runtimeConfig())
	if err != nil {
		return nil, err
	}

	stream, err := rt.NewStream(cfg.streamConfig())
	if err != nil {
		_ = rt.Destroy()
		return nil, err
	}

	return &Detector{
		runtime: rt,
		stream:  stream,
	}, nil
}

func (d *Detector) Detect(pcm []float32) ([]Segment, error) {
	if d == nil || d.stream == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	return d.stream.Detect(pcm)
}

func (d *Detector) Infer(samples []float32) (float32, error) {
	if d == nil || d.stream == nil {
		return 0, fmt.Errorf("invalid nil detector")
	}

	return d.stream.Infer(samples)
}

func (d *Detector) Reset() error {
	if d == nil || d.stream == nil {
		return fmt.Errorf("invalid nil detector")
	}

	return d.stream.Reset()
}

func (d *Detector) SetThreshold(value float32) {
	if d == nil || d.stream == nil {
		return
	}

	d.stream.SetThreshold(value)
}

func (d *Detector) Destroy() error {
	if d == nil || d.runtime == nil {
		return fmt.Errorf("invalid nil detector")
	}

	return d.runtime.Destroy()
}
