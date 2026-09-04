from collections import deque, defaultdict
import math
import time

class SimpleTracker:
    def __init__(self, max_lost=15, dist_thresh=60):
        self.next_id = 0
        self.objects = dict()    # id -> (x,y)
        self.lost = dict()       # id -> lost frames
        self.trajectories = defaultdict(lambda: deque(maxlen=40))  # id -> deque of (timestamp, (x,y))
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def update(self, detections):
        # detections: list of (x,y)
        if len(self.objects) == 0:
            for c in detections:
                self.objects[self.next_id] = c
                self.lost[self.next_id] = 0
                self.trajectories[self.next_id].append((time.time(), c))
                self.next_id += 1
            return self.objects

        unmatched_ids = set(self.objects.keys())
        assigned = {}
        # simple greedy nearest neighbor
        for det in detections:
            best_id = None
            best_dist = None
            for oid in list(unmatched_ids):
                ox, oy = self.objects[oid]
                d = math.hypot(det[0]-ox, det[1]-oy)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_id = oid
            if best_id is not None and best_dist <= self.dist_thresh:
                self.objects[best_id] = det
                self.lost[best_id] = 0
                self.trajectories[best_id].append((time.time(), det))
                unmatched_ids.remove(best_id)
                assigned[best_id] = det
            else:
                # new id
                self.objects[self.next_id] = det
                self.lost[self.next_id] = 0
                self.trajectories[self.next_id].append((time.time(), det))
                self.next_id += 1
        # increase lost for unmatched
        for oid in unmatched_ids:
            self.lost[oid] += 1
        # remove too-lost
        to_del = [oid for oid, l in self.lost.items() if l > self.max_lost]
        for oid in to_del:
            del self.objects[oid]
            del self.lost[oid]
            if oid in self.trajectories:
                del self.trajectories[oid]
        return self.objects

    def get_recent_movement(
        self,
        oid,
        seconds=5.0
    ):

        traj = self.trajectories.get(
            oid,
            None
        )

        if not traj:
            return 0.0

        now = time.time()

        pts = [
            p
            for (t, p) in traj
            if now - t <= seconds
        ]

        if len(pts) < 2:
            return 0.0

        total_dist = 0.0

        for i in range(2, len(pts), 2):

            total_dist += math.hypot(
                pts[i][0] - pts[i - 1][0],
                pts[i][1] - pts[i - 1][1]
            )

        return total_dist