from collections import defaultdict, deque
from dataclasses import dataclass
import math
import time


@dataclass
class AnomalyResult:
    animal_id: int
    score: float
    reasons: list
    metrics: dict
    message: str


class AnimalBehaviorAnomalyDetector:
    """Rolling, per-animal anomaly detection for movement and zone behavior."""

    def __init__(
        self,
        window_seconds=300.0,
        window_options=(60.0, 180.0, 300.0),
        min_samples=30,
        min_animals=4,
        movement_ratio_threshold=1.25,
        movement_min_delta=35.0,
        ratio_delta_threshold=0.8,
        z_threshold=2.8,
        min_sample_interval=1.0,
        confirmation_samples=20,
        ratio_extreme_low=0.1,
        ratio_extreme_high=0.9,
    ):
        self.window_options = tuple(sorted(float(w) for w in window_options))
        self.window_seconds = max(float(window_seconds), max(self.window_options))
        self.min_samples = min_samples
        self.min_animals = min_animals
        self.movement_ratio_threshold = movement_ratio_threshold
        self.movement_min_delta = movement_min_delta
        self.ratio_delta_threshold = ratio_delta_threshold
        self.z_threshold = z_threshold
        self.min_sample_interval = min_sample_interval
        self.confirmation_samples = confirmation_samples
        self.ratio_extreme_low = ratio_extreme_low
        self.ratio_extreme_high = ratio_extreme_high
        self.samples = defaultdict(deque)
        self.candidate_hits = defaultdict(int)
        self.confirmed_anomalies = {}
        self.latest_summaries = {}

    def reset(self):
        self.samples.clear()
        self.candidate_hits.clear()
        self.confirmed_anomalies.clear()
        self.latest_summaries.clear()

    def update(self, animal_states, now=None):
        """
        Store the latest animal states and return anomalies keyed by animal id.

        animal_states item format:
        {
            "id": int,
            "movement": float,
            "at_feeder": bool,
            "at_drinker": bool,
            "is_sleeping": bool
        }
        """
        now = time.time() if now is None else now
        active_ids = set()
        sampled = False

        for state in animal_states:
            animal_id = int(state["id"])
            active_ids.add(animal_id)
            animal_samples = self.samples[animal_id]
            if animal_samples and now - animal_samples[-1]["t"] < self.min_sample_interval:
                continue
            sampled = True
            animal_samples.append({
                "t": now,
                "movement": float(state.get("movement", 0.0)),
                "feeder": 1.0 if state.get("at_feeder", False) else 0.0,
                "drinker": 1.0 if state.get("at_drinker", False) else 0.0,
                "sleeping": 1.0 if state.get("is_sleeping", False) else 0.0,
            })

        self._trim_old_samples(now)

        for animal_id in list(self.candidate_hits.keys()):
            if animal_id not in active_ids:
                del self.candidate_hits[animal_id]
        for animal_id in list(self.confirmed_anomalies.keys()):
            if animal_id not in active_ids:
                del self.confirmed_anomalies[animal_id]

        if not sampled:
            return dict(self.confirmed_anomalies)

        summaries = {}
        for animal_id in active_ids:
            animal_samples = self.samples.get(animal_id, [])
            summary = self._best_window_summary(animal_samples, now)
            if summary is not None:
                summaries[animal_id] = summary

        if len(summaries) < self.min_animals:
            self.candidate_hits.clear()
            self.confirmed_anomalies.clear()
            self.latest_summaries = summaries
            return {}

        self.latest_summaries = summaries
        for animal_id in list(self.candidate_hits.keys()):
            if animal_id not in summaries:
                del self.candidate_hits[animal_id]
                self.confirmed_anomalies.pop(animal_id, None)

        raw_anomalies = {}
        for animal_id, metrics in summaries.items():
            result = self._check_animal(animal_id, metrics, list(summaries.values()))
            if result is not None:
                raw_anomalies[animal_id] = result

        for animal_id in summaries:
            if animal_id in raw_anomalies:
                self.candidate_hits[animal_id] += 1
                if self.candidate_hits[animal_id] >= self.confirmation_samples:
                    self.confirmed_anomalies[animal_id] = raw_anomalies[animal_id]
            else:
                self.candidate_hits[animal_id] = 0
                self.confirmed_anomalies.pop(animal_id, None)

        return dict(self.confirmed_anomalies)

    def _trim_old_samples(self, now):
        cutoff = now - self.window_seconds
        for animal_id in list(self.samples.keys()):
            animal_samples = self.samples[animal_id]
            while animal_samples and animal_samples[0]["t"] < cutoff:
                animal_samples.popleft()
            if not animal_samples:
                del self.samples[animal_id]

    def _best_window_summary(self, samples, now):
        best_summary = None
        best_error = None

        for window_seconds in self.window_options:
            cutoff = now - window_seconds
            window_samples = [
                sample
                for sample in samples
                if sample["t"] >= cutoff
            ]
            if len(window_samples) < self.min_samples:
                continue

            summary = self._summarize(window_samples, window_seconds)
            error = self._estimate_error(summary)
            if best_error is None or error < best_error:
                best_error = error
                best_summary = summary

        return best_summary

    def _summarize(self, samples, window_seconds):
        movement_values = [s["movement"] for s in samples]
        feeder_values = [s["feeder"] for s in samples]
        drinker_values = [s["drinker"] for s in samples]
        sleeping_values = [s["sleeping"] for s in samples]
        sample_count = len(samples)
        return {
            "movement": self._mean(movement_values),
            "movement_std": self._std(movement_values),
            "feeder_ratio": self._mean(feeder_values),
            "drinker_ratio": self._mean(drinker_values),
            "sleeping_ratio": self._mean(sleeping_values),
            "sample_count": sample_count,
            "window_seconds": window_seconds,
        }

    def _estimate_error(self, summary):
        sample_count = max(1, summary["sample_count"])
        movement_center = max(abs(summary["movement"]), self.movement_min_delta)
        movement_error = (
            summary["movement_std"]
            / math.sqrt(sample_count)
            / movement_center
        )

        ratio_errors = []
        for metric_name in ("feeder_ratio", "drinker_ratio", "sleeping_ratio"):
            ratio = summary[metric_name]
            ratio_errors.append(math.sqrt(max(0.0, ratio * (1.0 - ratio)) / sample_count))

        window_penalty = 1.0 / max(1.0, summary["window_seconds"])
        return movement_error + sum(ratio_errors) + window_penalty

    def _check_animal(self, animal_id, metrics, population):
        checks = [
            ("movement", "movement"),
            ("feeder_ratio", "ratio"),
            ("drinker_ratio", "ratio"),
            ("sleeping_ratio", "ratio"),
        ]
        reasons = []
        score = 0.0

        for metric_name, metric_type in checks:
            population_values = [m[metric_name] for m in population]
            if not population_values:
                continue

            value = metrics[metric_name]
            center = self._median(population_values)
            diff = value - center
            abs_diff = abs(diff)
            mad = self._median(abs(population_value - center) for population_value in population_values)
            robust_std = mad * 1.4826
            z_score = abs_diff / robust_std if robust_std > 0.000001 else 0.0

            if metric_type == "movement":
                denominator = max(abs(center), self.movement_min_delta)
                relative_diff = abs_diff / denominator
                is_anomaly = (
                    abs_diff >= self.movement_min_delta
                    and (
                        relative_diff >= self.movement_ratio_threshold
                        or z_score >= self.z_threshold
                    )
                )
            else:
                is_extreme_high = value >= self.ratio_extreme_high and diff > 0
                is_extreme_low = value <= self.ratio_extreme_low and diff < 0
                is_anomaly = (
                    abs_diff >= self.ratio_delta_threshold
                    and (is_extreme_high or is_extreme_low)
                    and z_score >= self.z_threshold
                )

            if is_anomaly:
                reasons.append(self._reason_text(metric_name, diff))
                score += min(3.0, z_score if math.isfinite(z_score) else 3.0)

        if not reasons:
            return None

        message = f"#{animal_id}: " + ", ".join(reasons[:2])
        return AnomalyResult(
            animal_id=animal_id,
            score=score,
            reasons=reasons,
            metrics=metrics,
            message=message,
        )

    def _reason_text(self, metric_name, diff):
        more = diff > 0
        if metric_name == "movement":
            return "szokatlanul sokat mozog" if more else "szokatlanul keveset mozog"
        if metric_name == "feeder_ratio":
            return "az atlagnal tobbet van az etetonel" if more else "az atlagnal kevesebbet van az etetonel"
        if metric_name == "drinker_ratio":
            return "az atlagnal tobbet van az itatonal" if more else "az atlagnal kevesebbet van az itatonal"
        if metric_name == "sleeping_ratio":
            return "az atlagnal tobbet alszik" if more else "az atlagnal kevesebbet alszik"
        return "elter az atlagtol"

    def _mean(self, values):
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _median(self, values):
        values = sorted(values)
        if not values:
            return 0.0
        middle = len(values) // 2
        if len(values) % 2 == 1:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    def _std(self, values):
        values = list(values)
        if len(values) < 2:
            return 0.0
        avg = self._mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)
