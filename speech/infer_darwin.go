//go:build darwin

package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func (s *Stream) Infer(samples []float32) (float32, error) {
	if s == nil {
		return 0, fmt.Errorf("invalid nil stream")
	}
	if len(samples) < contextLen {
		return 0, fmt.Errorf("not enough samples")
	}

	pcm := samples
	if s.hasContext {
		pcm = make([]float32, 0, contextLen+len(samples))
		pcm = append(pcm, s.ctx[:]...)
		pcm = append(pcm, samples...)
	}
	// Save the last contextLen samples as context for the next iteration.
	copy(s.ctx[:], samples[len(samples)-contextLen:])
	s.hasContext = true

	rt := s.runtime
	session, err := rt.acquireSession()
	if err != nil {
		return 0, err
	}
	defer rt.releaseSession(session)

	// Create tensors
	var pcmValue *C.OrtValue
	pcmInputDims := []C.longlong{
		1,
		C.longlong(len(pcm)),
	}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(rt.api, rt.memoryInfo, unsafe.Pointer(&pcm[0]), C.size_t(len(pcm)*4), &pcmInputDims[0], C.size_t(len(pcmInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pcmValue)
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}
	defer C.OrtApiReleaseValue(rt.api, pcmValue)

	var stateValue *C.OrtValue
	stateNodeInputDims := []C.longlong{2, 1, 128}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(rt.api, rt.memoryInfo, unsafe.Pointer(&s.state[0]), C.size_t(stateLen*4), &stateNodeInputDims[0], C.size_t(len(stateNodeInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &stateValue)
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}
	defer C.OrtApiReleaseValue(rt.api, stateValue)

	var rateValue *C.OrtValue
	rateInputDims := []C.longlong{1}
	rate := []C.int64_t{C.int64_t(s.cfg.SampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(rt.api, rt.memoryInfo, unsafe.Pointer(&rate[0]), C.size_t(8), &rateInputDims[0], C.size_t(len(rateInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &rateValue)
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}
	defer C.OrtApiReleaseValue(rt.api, rateValue)

	// Run inference
	inputs := []*C.OrtValue{pcmValue, stateValue, rateValue}
	outputs := []*C.OrtValue{nil, nil}

	inputNames := []*C.char{
		rt.cStrings["input"],
		rt.cStrings["state"],
		rt.cStrings["sr"],
	}
	outputNames := []*C.char{
		rt.cStrings["output"],
		rt.cStrings["stateN"],
	}
	status = C.OrtApiRun(rt.api, session, nil, &inputNames[0], &inputs[0], C.size_t(len(inputNames)), &outputNames[0], C.size_t(len(outputNames)), &outputs[0])
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}
	defer C.OrtApiReleaseValue(rt.api, outputs[0])
	defer C.OrtApiReleaseValue(rt.api, outputs[1])

	// Get output values from tensor data
	var prob unsafe.Pointer
	var stateN unsafe.Pointer

	status = C.OrtApiGetTensorMutableData(rt.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(rt.api, outputs[1], &stateN)
	defer C.OrtApiReleaseStatus(rt.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(rt.api, status)))
	}

	speechProb := *(*float32)(prob)
	C.memcpy(unsafe.Pointer(&s.state[0]), stateN, stateLen*4)

	// Return speech probability
	return speechProb, nil
}
