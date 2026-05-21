#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>

#include "ort_bridge.h"

const OrtApi* OrtGetApi() {
  // This bridge only uses APIs available in v17; request that version so newer
  // headers can still run with older compatible runtime DLLs.
  return OrtGetApiBase()->GetApi(17);
}

void OrtApiReleaseStatus(OrtApi* api, OrtStatus* status) {
  return api->ReleaseStatus(status);
}

const char* OrtApiGetErrorMessage(OrtApi* api, OrtStatus* status) {
  return api->GetErrorMessage(status);
}

OrtStatus* OrtApiCreateEnv(OrtApi* api, OrtLoggingLevel log_level, const char* log_id, OrtEnv** env) {
  return api->CreateEnv(log_level, log_id, env);
}

void OrtApiReleaseEnv(OrtApi* api, OrtEnv* env) {
  return api->ReleaseEnv(env);
}

OrtStatus* OrtApiCreateSessionOptions(OrtApi* api, OrtSessionOptions** opts) {
  return api->CreateSessionOptions(opts);
}

void OrtApiReleaseSessionOptions(OrtApi* api, OrtSessionOptions* opts) {
  return api->ReleaseSessionOptions(opts);
}

OrtStatus* OrtApiSetIntraOpNumThreads(OrtApi* api, OrtSessionOptions* opts, int intra_op_num_threads) {
  return api->SetIntraOpNumThreads(opts, intra_op_num_threads);
}

OrtStatus* OrtApiSetInterOpNumThreads(OrtApi* api, OrtSessionOptions* opts, int inter_op_num_threads) {
  return api->SetInterOpNumThreads(opts, inter_op_num_threads);
}

OrtStatus* OrtApiSetSessionGraphOptimizationLevel(OrtApi* api, OrtSessionOptions* opts, GraphOptimizationLevel graph_optimization_level) {
  return api->SetSessionGraphOptimizationLevel(opts, graph_optimization_level);
}

OrtStatus* OrtApiCreateSessionFromArray(OrtApi* api, OrtEnv* env, const void* model_data, size_t model_data_len,
    OrtSessionOptions* opts, OrtSession** session) {
  return api->CreateSessionFromArray(env, model_data, model_data_len, opts, session);
}

void OrtApiReleaseSession(OrtApi* api, OrtSession* session) {
  return api->ReleaseSession(session);
}

OrtStatus* OrtApiCreateCpuMemoryInfo(OrtApi* api, enum OrtAllocatorType alloc_type, enum OrtMemType mem_type, OrtMemoryInfo** minfo) {
  return api->CreateCpuMemoryInfo(alloc_type, mem_type, minfo);
}

void OrtApiReleaseMemoryInfo(OrtApi* api, OrtMemoryInfo *minfo) {
  return api->ReleaseMemoryInfo(minfo);
}

OrtStatus* OrtApiCreateTensorWithDataAsOrtValue(OrtApi* api, const OrtMemoryInfo* minfo, void* data,
    size_t data_len, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType data_type, OrtValue** value) {
  return api->CreateTensorWithDataAsOrtValue(minfo, data, data_len, shape, shape_len, data_type, value);
}

void OrtApiReleaseValue(OrtApi* api, OrtValue *value) {
  return api->ReleaseValue(value);
}

OrtStatus* OrtApiRun(OrtApi* api, OrtSession* session, const OrtRunOptions* run_options,
    const char* const* input_names, const OrtValue* const* inputs, size_t inputs_len,
    const char* const* output_names, size_t output_names_len, OrtValue** outputs) {
  return api->Run(session, run_options, input_names, inputs, inputs_len, output_names, output_names_len, outputs);
}

OrtStatus* OrtApiGetTensorMutableData(OrtApi* api, OrtValue* value, void** data) {
  return api->GetTensorMutableData(value, data);
}
