155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
"""
            try:
                self._model.cpu()
                self.device = "cpu"
                logger.info("Offloaded model %s:%s to CPU", self.name, self.version)
            except Exception:
                logger.exception("cpu_offload failed")


class ONNXModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str, name: str, version: str, use_gpu: bool = False):
        super().__init__(name, version)
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not available")
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._sess = None
        self._loaded = False

    def load(self):
        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            # depending on platform, use CUDAExecutionProvider or others
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Creating ONNXRuntime session %s:%s providers=%s", self.name, self.version, providers)
        self._sess = ort.InferenceSession(self.model_path, providers=providers)
        self._loaded = True

    def warmup(self, sample_input: Dict[str, Any], iters: int = 1):
        if not self._loaded:
            self.load()
        logger.info("Warming ONNX model %s:%s", self.name, self.version)
        for _ in range(iters):
            try:
                in_names = [n.name for n in self._sess.get_inputs()]
                feed = {}
                # naive mapping: sample_input must be dict of input_name -> numpy array
                for nm in in_names:
                    if nm in sample_input:
                        feed[nm] = sample_input[nm]
                self._sess.run(None, feed)
            except Exception:
                logger.exception("onnx warmup failed")

    def predict_batch(self, batched_inputs: list):
        if not self._loaded:
            self.load()
        outputs = []
        in_names = [n.name for n in self._sess.get_inputs()]
        # batched_inputs: list of dicts keyed by input names
        for inp in batched_inputs:
            try:
                feed = {k: v for k, v in inp.items() if k in in_names}
                out = self._sess.run(None, feed)
                outputs.append(out)
            except Exception as exc:
                logger.exception("onnx inference error: %s", exc)
                outputs.append({"error": str(exc)})
        return outputs
api/model_loader.py
