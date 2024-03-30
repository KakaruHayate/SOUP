import numpy as np
import onnx
import onnxruntime
import math
import pyworld as pw


def infer(model_path,input_data):
    ort_session = onnxruntime.InferenceSession(model_path)
    inputs = ort_session.get_inputs()
    input_names=[]
    for i in inputs:
        input_names.append(i.name)
    output_name = ort_session.get_outputs()[0].name
    output_names=[]
    for i in ort_session.get_outputs():
        output_names.append(i.name)
    output_data = ort_session.run(output_names, input_data)
    res={}
    for i in range(len(output_names)):
        res.update({output_names[i]:output_data[i]})
    return res


class SOME:
    def __init__(self,some_path):
        self.some_path=str(some_path)
    def inference(self,wf,osr=44100):
        wf = wf.astype(np.float32)
        SOME_responses=[]
        slices=[{"waveform":wf,"offset":0}]
        for segment in slices:
            segment_wf=segment["waveform"]
            res=infer(f"{self.some_path}",{
                "waveform":segment_wf.reshape([1,segment_wf.shape[0]])
            })
            SOME_responses.append({"waveform":res,"offset":segment["offset"]})
        for rd in SOME_responses:
            res=rd["waveform"]

        return res

def expand_midi(res, output_array):
    note_midi = res['note_midi'][0]
    note_rest = res['note_rest'][0]
    note_dur = res['note_dur'][0]

    note_midi = np.where(note_rest, 0, note_midi)
    note_dur = note_dur / np.sum(note_dur)
    output_length = len(output_array)
    repeats = np.round(note_dur * output_length).astype(int)
    expanded_midi = np.repeat(note_midi, repeats)
    if len(expanded_midi) > output_length:
        expanded_midi = expanded_midi[:output_length]
    elif len(expanded_midi) < output_length:
        expanded_midi = np.pad(expanded_midi, (0, output_length - len(expanded_midi)), 'edge')

    return expanded_midi
