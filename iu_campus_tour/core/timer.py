import time


class Timer:
    def __init__(self, max_dt=0.1, fps_interval=0.5):
        self.max_dt = max_dt            
        self.fps_interval = fps_interval
        self.dt = 0.0                   
        self.total = 0.0                
        self.fps = 0.0
        self.fps_updated = False       
        self._last = time.perf_counter()  
        self._acc = 0.0
        self._frames = 0

    def tick(self):
        now = time.perf_counter()
        raw = now - self._last
        self._last = now

        self.dt = min(raw, self.max_dt)
        self.total += self.dt

        self._acc += raw
        self._frames += 1
        self.fps_updated = False
        if self._acc >= self.fps_interval:
            self.fps = self._frames / self._acc
            self._acc, self._frames = 0.0, 0
            self.fps_updated = True
        return self.dt