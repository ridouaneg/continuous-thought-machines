"""CTM for repetition counting.

Thin subclass of ContinuousThoughtMachineVideo. The only semantic
difference is that out_dims encodes count buckets rather than action
classes. The frame-coupling forward loop is identical: at internal tick t,
the model attends to frame t // iterations_per_frame.

The CTM paper claims neurons learn to oscillate at frequencies induced by
the input. For periodic videos (repetitive actions) this predicts that the
dominant per-neuron FFT frequency scales with the repetition count — a
directly testable hypothesis via the analysis notebook.
"""

from tasks.video.model import ContinuousThoughtMachineVideo


class ContinuousThoughtMachineRepCount(ContinuousThoughtMachineVideo):
    """CTM for repetition counting via bucket classification.

    Each output dimension corresponds to a count bucket (0, 1, ..., n_count_buckets-1).
    The frame-coupling tick loop is inherited from ContinuousThoughtMachineVideo
    unchanged — the only difference is the semantics of out_dims.

    Args:
        n_frames: Number of video frames sampled per clip.
        iterations_per_frame: CTM internal ticks spent on each frame.
            Total iterations = n_frames * iterations_per_frame.
        All other args are forwarded to ContinuousThoughtMachine via
        ContinuousThoughtMachineVideo.
    """
    pass
