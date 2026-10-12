"""Control-flow exceptions shared across the experiment."""


class QuitExperiment(Exception):
    pass


class RecalibrateRequest(Exception):
    pass


class StimulusError(Exception):
    pass
