package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"os"
	"sync"
	"unsafe"
)

type LogLevel int

func (l LogLevel) OrtLoggingLevel() C.OrtLoggingLevel {
	switch l {
	case LevelVerbose:
		return C.ORT_LOGGING_LEVEL_VERBOSE
	case LogLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO
	case LogLevelWarn:
		return C.ORT_LOGGING_LEVEL_WARNING
	case LogLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR
	case LogLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL
	default:
		return C.ORT_LOGGING_LEVEL_WARNING
	}
}

const (
	LevelVerbose LogLevel = iota + 1
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

type RuntimeConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The loglevel for the onnx environment, by default it is set to LogLevelWarn.
	LogLevel LogLevel
	// The number of ONNX Runtime sessions to load and share across streams.
	NumSessions int
	// The number of intra-op threads for each ONNX Runtime session. Defaults to 1.
	IntraOpNumThreads int
	// The number of inter-op threads for each ONNX Runtime session. Defaults to 1.
	InterOpNumThreads int
}

func (c RuntimeConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if c.NumSessions < 0 {
		return fmt.Errorf("invalid NumSessions: should be a positive number")
	}

	if c.IntraOpNumThreads < 0 {
		return fmt.Errorf("invalid IntraOpNumThreads: should be a positive number")
	}

	if c.InterOpNumThreads < 0 {
		return fmt.Errorf("invalid InterOpNumThreads: should be a positive number")
	}

	return nil
}

func (c RuntimeConfig) withDefaults() RuntimeConfig {
	if c.NumSessions == 0 {
		c.NumSessions = 1
	}
	if c.IntraOpNumThreads == 0 {
		c.IntraOpNumThreads = 1
	}
	if c.InterOpNumThreads == 0 {
		c.InterOpNumThreads = 1
	}
	return c
}

// Runtime owns ONNX Runtime resources and a small pool of reusable sessions.
type Runtime struct {
	api          *C.OrtApi
	env          *C.OrtEnv
	sessionOpts  *C.OrtSessionOptions
	memoryInfo   *C.OrtMemoryInfo
	cStrings     map[string]*C.char
	sessions     []*C.OrtSession
	pool         chan *C.OrtSession
	modelData    unsafe.Pointer
	modelDataLen C.size_t

	mu        sync.RWMutex
	destroyed bool
}

func NewRuntime(cfg RuntimeConfig) (*Runtime, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	cfg = cfg.withDefaults()

	modelData, err := os.ReadFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model: %w", err)
	}
	if len(modelData) == 0 {
		return nil, fmt.Errorf("failed to read model: empty model file")
	}
	modelDataPtr := C.CBytes(modelData)
	if modelDataPtr == nil {
		return nil, fmt.Errorf("failed to allocate model data")
	}

	rt := &Runtime{
		cStrings:     map[string]*C.char{},
		sessions:     make([]*C.OrtSession, 0, cfg.NumSessions),
		pool:         make(chan *C.OrtSession, cfg.NumSessions),
		modelData:    modelDataPtr,
		modelDataLen: C.size_t(len(modelData)),
	}

	rt.api = C.OrtGetApi()
	if rt.api == nil {
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to get API")
	}

	rt.cStrings["loggerName"] = C.CString("vad")
	status := C.OrtApiCreateEnv(rt.api, cfg.LogLevel.OrtLoggingLevel(), rt.cStrings["loggerName"], &rt.env)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to create env: %s", msg)
	}

	status = C.OrtApiCreateSessionOptions(rt.api, &rt.sessionOpts)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to create session options: %s", msg)
	}

	status = C.OrtApiSetIntraOpNumThreads(rt.api, rt.sessionOpts, C.int(cfg.IntraOpNumThreads))
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to set intra threads: %s", msg)
	}

	status = C.OrtApiSetInterOpNumThreads(rt.api, rt.sessionOpts, C.int(cfg.InterOpNumThreads))
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to set inter threads: %s", msg)
	}

	status = C.OrtApiSetSessionGraphOptimizationLevel(rt.api, rt.sessionOpts, C.ORT_ENABLE_ALL)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to set session graph optimization level: %s", msg)
	}

	for i := 0; i < cfg.NumSessions; i++ {
		var session *C.OrtSession
		status = C.OrtApiCreateSessionFromArray(rt.api, rt.env, rt.modelData, rt.modelDataLen, rt.sessionOpts, &session)
		if status != nil {
			msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
			C.OrtApiReleaseStatus(rt.api, status)
			_ = rt.Destroy()
			return nil, fmt.Errorf("failed to create session: %s", msg)
		}
		rt.sessions = append(rt.sessions, session)
		rt.pool <- session
	}

	status = C.OrtApiCreateCpuMemoryInfo(rt.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &rt.memoryInfo)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(rt.api, status))
		C.OrtApiReleaseStatus(rt.api, status)
		_ = rt.Destroy()
		return nil, fmt.Errorf("failed to create memory info: %s", msg)
	}

	rt.cStrings["input"] = C.CString("input")
	rt.cStrings["sr"] = C.CString("sr")
	rt.cStrings["state"] = C.CString("state")
	rt.cStrings["stateN"] = C.CString("stateN")
	rt.cStrings["output"] = C.CString("output")

	return rt, nil
}

func (rt *Runtime) Destroy() error {
	if rt == nil {
		return fmt.Errorf("invalid nil runtime")
	}

	rt.mu.Lock()
	if rt.destroyed {
		rt.mu.Unlock()
		return nil
	}
	rt.destroyed = true
	if rt.pool != nil {
		close(rt.pool)
	}
	rt.mu.Unlock()

	if rt.api != nil {
		for _, session := range rt.sessions {
			if session != nil {
				C.OrtApiReleaseSession(rt.api, session)
			}
		}
		if rt.memoryInfo != nil {
			C.OrtApiReleaseMemoryInfo(rt.api, rt.memoryInfo)
		}
		if rt.sessionOpts != nil {
			C.OrtApiReleaseSessionOptions(rt.api, rt.sessionOpts)
		}
		if rt.env != nil {
			C.OrtApiReleaseEnv(rt.api, rt.env)
		}
	}

	for _, ptr := range rt.cStrings {
		if ptr != nil {
			C.free(unsafe.Pointer(ptr))
		}
	}
	if rt.modelData != nil {
		C.free(rt.modelData)
		rt.modelData = nil
	}

	return nil
}

func (rt *Runtime) acquireSession() (*C.OrtSession, error) {
	if rt == nil {
		return nil, fmt.Errorf("invalid nil runtime")
	}

	rt.mu.RLock()
	if rt.destroyed || rt.pool == nil {
		rt.mu.RUnlock()
		return nil, fmt.Errorf("runtime is destroyed")
	}
	pool := rt.pool
	rt.mu.RUnlock()

	session, ok := <-pool
	if !ok || session == nil {
		return nil, fmt.Errorf("runtime is destroyed")
	}

	return session, nil
}

func (rt *Runtime) releaseSession(session *C.OrtSession) {
	if rt == nil || session == nil {
		return
	}

	rt.mu.RLock()
	defer rt.mu.RUnlock()

	if rt.destroyed || rt.pool == nil {
		return
	}

	rt.pool <- session
}
