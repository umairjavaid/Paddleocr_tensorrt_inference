import ctypes
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
batch_size = 1

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context

#read tensorrt
# def set_up(engine_file_path):
#     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             engine = runtime.deserialize_cuda_engine(f.read())
#             context = engine.create_execution_context()

def set_up(engine, context):
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            h_input = None
            input_shape = engine.get_binding_shape(binding)
            print("input_shape: ", input_shape)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float16).itemsize  # in bytes
            d_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            print("output_shape: ", output_shape)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float16)
            d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return engine, context, h_input, h_output, d_input, d_output, stream, output_shape

def preprocess_image(img):
    input_data = np.array(img, dtype=np.float32)
    return input_data

onnx_file = "sast_model_896_1536.onnx"
engine, context = build_engine(onnx_file)
engine, context, h_input, h_output, d_input, d_output, stream, output_shape = set_up(engine, context)

def infer(img, engine, device_input,  host_input, stream, context, device_output, host_output, output_shape):
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    return host_output