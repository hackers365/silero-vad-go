package speech

import (
	"fmt"
	"log/slog"
)

const (
	stateLen   = 2 * 1 * 128
	contextLen = 64
)

type StreamConfig struct {
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence to wait for each speech segment before separating it.
	MinSilenceDurationMs int
	// The padding to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
}

func (c StreamConfig) IsValid() error {
	if c.SampleRate != 8000 && c.SampleRate != 16000 {
		return fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if c.Threshold <= 0 || c.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}

	if c.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be a positive number")
	}

	if c.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be a positive number")
	}

	return nil
}

// Stream keeps the per-audio-stream Silero and segmentation state.
type Stream struct {
	runtime *Runtime
	cfg     StreamConfig

	state [stateLen]float32
	ctx   [contextLen]float32

	hasContext bool
	currSample int
	triggered  bool
	tempEnd    int
}

func (rt *Runtime) NewStream(cfg StreamConfig) (*Stream, error) {
	if rt == nil {
		return nil, fmt.Errorf("invalid nil runtime")
	}

	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	rt.mu.RLock()
	destroyed := rt.destroyed
	rt.mu.RUnlock()
	if destroyed {
		return nil, fmt.Errorf("runtime is destroyed")
	}

	return &Stream{
		runtime: rt,
		cfg:     cfg,
	}, nil
}

// Segment contains timing information of a speech segment.
type Segment struct {
	// The relative timestamp in seconds of when a speech segment begins.
	SpeechStartAt float64
	// The relative timestamp in seconds of when a speech segment ends.
	SpeechEndAt float64
}

func (s *Stream) Detect(pcm []float32) ([]Segment, error) {
	if s == nil {
		return nil, fmt.Errorf("invalid nil stream")
	}

	windowSize := 512
	if s.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return nil, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection", slog.Int("samplesLen", len(pcm)))

	minSilenceSamples := s.cfg.MinSilenceDurationMs * s.cfg.SampleRate / 1000
	speechPadSamples := s.cfg.SpeechPadMs * s.cfg.SampleRate / 1000

	var segments []Segment
	for i := 0; i < len(pcm)-windowSize; i += windowSize {
		speechProb, err := s.Infer(pcm[i : i+windowSize])
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		s.currSample += windowSize

		if speechProb >= s.cfg.Threshold && s.tempEnd != 0 {
			s.tempEnd = 0
		}

		if speechProb >= s.cfg.Threshold && !s.triggered {
			s.triggered = true
			speechStartAt := (float64(s.currSample-windowSize-speechPadSamples) / float64(s.cfg.SampleRate))

			if speechStartAt < 0 {
				speechStartAt = 0
			}

			slog.Debug("speech start", slog.Float64("startAt", speechStartAt))
			segments = append(segments, Segment{
				SpeechStartAt: speechStartAt,
			})
		}

		if speechProb < (s.cfg.Threshold-0.15) && s.triggered {
			if s.tempEnd == 0 {
				s.tempEnd = s.currSample
			}

			if s.currSample-s.tempEnd < minSilenceSamples {
				continue
			}

			speechEndAt := (float64(s.tempEnd+speechPadSamples) / float64(s.cfg.SampleRate))
			s.tempEnd = 0
			s.triggered = false
			slog.Debug("speech end", slog.Float64("endAt", speechEndAt))

			if len(segments) < 1 {
				return nil, fmt.Errorf("unexpected speech end")
			}

			segments[len(segments)-1].SpeechEndAt = speechEndAt
		}
	}

	slog.Debug("speech detection done", slog.Int("segmentsLen", len(segments)))

	return segments, nil
}

func (s *Stream) Reset() error {
	if s == nil {
		return fmt.Errorf("invalid nil stream")
	}

	s.currSample = 0
	s.triggered = false
	s.tempEnd = 0
	s.hasContext = false
	for i := 0; i < stateLen; i++ {
		s.state[i] = 0
	}
	for i := 0; i < contextLen; i++ {
		s.ctx[i] = 0
	}

	return nil
}

func (s *Stream) SetThreshold(value float32) {
	s.cfg.Threshold = value
}
