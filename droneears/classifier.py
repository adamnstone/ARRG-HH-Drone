"""Load the exported TFLite classifier as a callable: (47, 40) normalised log-mel window -> p(drone)."""
import numpy as np


def load_tflite_classifier(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]; outp = it.get_output_details()[0]
    s_in, z_in = inp["quantization"]; s_out, z_out = outp["quantization"]
    is_int8 = inp["dtype"] == np.int8

    def predict(window):
        x = np.asarray(window, np.float32)[None, ..., None]
        if is_int8:
            x = np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x); it.invoke()
        o = it.get_tensor(outp["index"])[0].astype(np.float32)
        return float((o[1] - z_out) * s_out) if is_int8 else float(o[1])
    predict.is_int8 = is_int8
    return predict
