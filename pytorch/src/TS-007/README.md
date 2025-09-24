# TS-007: Numpy-style Tensor Slicing (PyTorch)

This test suite validates common numpy-style slicing operations on PyTorch tensors. The focus is on simple and practical patterns frequently used in machine learning code.

What this suite covers:
- Simple use cases (basic row/column selections)
- Ranges (start:end:step, including open-ended and negative steps)
- Explicit start/end bounds and None usage
- First, last, and second-last selections
- Typical ML slicing on NCHW tensors (channel picking, spatial crops, time windows)

Use Cases (UC files):
- UC-001: Simple basics (rows/cols, single-dim slices)
- UC-002: Range slices with steps (positive/negative, open-ended)
- UC-003: Explicit start/end bounds and None
- UC-004: First/last/second-last, ellipsis
- UC-005: Typical ML slices (NCHW channel subset, crops, sequence window)

How to run:
- From the `pytorch` directory with your environment activated:
  - Run all suites: `gt src results`
  - Run this suite only: `gt src/TS-007 results`

Notes:
- All inputs have `requires_grad=True` to enable gradient tracing.
- Each test returns ([inputs], output) as required by gradienttracer.
