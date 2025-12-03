 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
"""
        self._task = self.loop.create_task(self._batcher_loop())

    async def submit(self, item: Any):
        fut = self.loop.create_future()
        await self._queue.put((item, fut))
        return await fut

    async def _batcher_loop(self):
        while not self._stopped:
            try:
                first = await self._queue.get()
                items = [first[0]]
                futures = [first[1]]
                start = time.time()
                # drain within max_latency_ms or until max_batch_size
                while (time.time() - start) * 1000.0 < self.max_latency_ms and len(items) < self.max_batch_size:
                    try:
                        item, fut = self._queue.get_nowait()
                        items.append(item)
                        futures.append(fut)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0)  # yield
                # Now process batch (call blocking function in threadpool)
                results = await self.loop.run_in_executor(None, self._safe_process, items)
                # results must be list-like with len == len(items)
                if not isinstance(results, (list, tuple)) or len(results) != len(items):
                    # best-effort: broadcast single result
                    for fut in futures:
                        if not fut.done():
                            fut.set_result(results)
                else:
                    for fut, res in zip(futures, results):
                        if not fut.done():
                            fut.set_result(res)
            except Exception:
                logger.exception("batcher loop error")
                await asyncio.sleep(0.1)

    def _safe_process(self, items: List[Any]):
        try:
            return self.process_batch(items)
        except Exception as exc:
            logger.exception("process_batch failed: %s", exc)
            # return list of errors
            return [{"error": str(exc)} for _ in items]

    async def stop(self):
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
