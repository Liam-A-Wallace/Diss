"""Eye-tracker stub — SeeSo integration goes here later."""
import os
import time


class GazeRecorder:
    def __init__(self, output_path, participant_id):
        self.output_path = output_path
        self.participant_id = participant_id
        self._events = []

    def start_session(self):
        self._events.append(self._event("session_start"))
        # TODO: SeeSo init + calibration

    def start_trial(self, trial_id):
        self._events.append(self._event("trial_start", trial_id=trial_id))
        # TODO: SeeSo trial onset timestamp

    def poll(self):
        # TODO: return latest gaze sample (x, y, confidence, t)
        return None

    def stop_trial(self):
        self._events.append(self._event("trial_stop"))
        # TODO: SeeSo flush trial buffer

    def stop_session(self):
        self._events.append(self._event("session_stop"))
        # TODO: SeeSo teardown
        self._write()

    def _event(self, name, trial_id=None):
        return {
            "participant": self.participant_id,
            "t": round(time.time(), 4),
            "event": name,
            "trial_id": trial_id or "",
        }

    def _write(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write("participant,t,event,trial_id\n")
            for e in self._events:
                f.write("{participant},{t},{event},{trial_id}\n".format(**e))
