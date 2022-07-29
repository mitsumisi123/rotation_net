
import numpy as np
import onnxruntime
import time
def main():
    onnx_path = "model/rotation.onnx"

    x = np.load("test_image/image.npy")
    print(onnxruntime.get_available_providers())
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['VsiNpuExecutionProvider'])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print("process time:", time.time()-start)
    print(ort_outs)
    # compare ONNX Runtime and PyTorch results

if __name__ == "__main__":
    main()