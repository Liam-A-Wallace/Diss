"""Spawn-safe worker configuration for the local, unmodified GazeFollower SDK."""
from functools import partial


def worker_entry(*args, model_path, lock_exposure, exposure):
    # Explicit initialization also works when demo.py is the spawn entry module.
    import gazefollower_patch
    gazefollower_patch.apply()
    from gazefollower.multiprocess import mp_worker
    factory = mp_worker.create_component

    def configured_component(comp_type, comp_kind, kwargs=None):
        options = dict(kwargs or {})
        if comp_kind == "gaze_estimator":
            options["model_path"] = model_path
        if comp_kind == "camera":
            from gazefollower.camera import WebCamCamera
            return WebCamCamera(webcam_id=options.get("webcam_id", 0),
                                lock_exposure=lock_exposure, exposure=exposure)
        return factory(comp_type, comp_kind, options)

    mp_worker.create_component = configured_component
    try:
        mp_worker.mp_worker_entry(*args)
    finally:
        mp_worker.create_component = factory


def configure_worker(model_path, lock_exposure, exposure):
    from gazefollower.multiprocess import mp_gazefollower
    mp_gazefollower.mp_worker_entry = partial(
        worker_entry, model_path=model_path, lock_exposure=lock_exposure, exposure=exposure)