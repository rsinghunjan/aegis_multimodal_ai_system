206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
"""
    # basic input validation: require at least one modality
    if not (req.text or req.image_base64 or req.image_url or req.audio_base64):
        raise HTTPException(status_code=400, detail="No input provided; supply text, image or audio")

    # decode base64 examples (don't store in memory in prod for large files)
    # TODO: stream decode if large
    if req.image_base64:
        try:
            _ = decode_base64_file(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid image_base64")

    if req.audio_base64:
        try:
            _ = decode_base64_file(req.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid audio_base64")

    # run model (placeholder)
    result = registry.predict(model_obj, req)
    elapsed_ms = (time.time() - start) * 1000.0
    resp = PredictionResponse(
        request_id=request_id,
        model=model_name,
        version=version,
        result=result,
        metrics={"inference_ms": elapsed_ms}
    )
    logger.info("predict %s %s request_id=%s elapsed=%.2fms", model_name, version, request_id, elapsed_ms)
    return resp


# --- Prediction endpoint (multipart file uploads) ----------------------------------------

@app.post("/v1/models/{model_name}/versions/{version}/predict-multipart", response_model=PredictionResponse)
async def predict_multipart(model_name: str, version: str, text: Optional[str] = None,
                            image_file: Optional[UploadFile] = File(None),
                            audio_file: Optional[UploadFile] = File(None),
                            auth=Depends(optional_auth)):
    request_id = make_request_id()
    start = time.time()
    try:
        model_obj = registry.load(model_name, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")

    # read uploaded files carefully; stream in production
    image_b64 = None
    audio_b64 = None
    if image_file:
        image_bytes = await image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    if audio_file:
        audio_bytes = await audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    req = PredictionRequest(text=text, image_base64=image_b64, audio_base64=audio_b64, parameters={})
    result = registry.predict(model_obj, req)
    elapsed_ms = (time.time() - start) * 1000.0
    return PredictionResponse(
        request_id=request_id, model=model_name, version=version, result=result,
        metrics={"inference_ms": elapsed_ms}
    )


# --- Basic error handler -----------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.status_code, "message": exc.detail})


# --- Local runner (uvicorn) --------------------------------------------------------------

if __name__ == "__main__":
