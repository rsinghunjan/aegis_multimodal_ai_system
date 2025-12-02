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
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
"""
    async def _drain_batch(self) -> List[Tuple[str, asyncio.Future]]:
        """
        Wait for at least one item, then collect up to max_batch_size items within max_latency_s.
        """
        items = []
        # block until at least one item is available
        item = await self._queue.get()
        items.append(item)
        start = time.time()
        # keep collecting until max_batch_size or timeout
        while len(items) < self.max_batch_size:
            elapsed = time.time() - start
            remaining = self.max_latency_s - elapsed
            if remaining <= 0:
                break
            try:
                # short wait for next item
                nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                items.append(nxt)
            except asyncio.TimeoutError:
                break
        return items

    async def _worker_loop(self):
        logger.info("BatchWorker started (max_batch_size=%d, max_latency_s=%.3f)", self.max_batch_size, self.max_latency_s)
        while not self._stop:
            try:
                batch = await self._drain_batch()
                texts = [t for (t, f) in batch]
                # Call model (sync or async). If predict_fn is coroutine, await it.
                t0 = time.time()
                try:
                    res = self.predict_fn(texts)
                    if asyncio.iscoroutine(res):
                        res = await res
                except Exception as e:
                    logger.exception("Batch predict failed: %s", e)
                    # set exception on futures
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                    continue
                # Expect res to be list-like with same length
                if not isinstance(res, (list, tuple)) or len(res) != len(batch):
                    err = RuntimeError("Batch predict returned invalid result length")
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(err)
                    continue

                # set results
                for (_, fut), out in zip(batch, res):
                    if not fut.done():
                        fut.set_result(out)
                latency = time.time() - t0
                logger.debug("Batch of %d processed in %.4f s", len(batch), latency)
            except Exception:
                logger.exception("BatchWorker loop error")
                await asyncio.sleep(0.1)

    def start(self):
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._stop = True
        # allow running task to exit gracefully
        if self._task:
            await self._task
