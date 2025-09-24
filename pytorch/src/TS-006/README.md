# TS-006: Simple Tensor Ops with Broadcasting (PyTorch)

This test suite validates simple elementwise tensor operations that rely on PyTorch broadcasting rules using NCHW layout with batched inputs.

What this suite covers:
- Addition (+) and subtraction (-)
- NCHW layout with batch dimension (B,C,H,W)
- Common broadcasting patterns:
  - Scalar -> (B,C,H,W)
  - Channel bias (1,C,1,1) -> (B,C,H,W)
  - Spatial map (1,1,H,W) -> (B,C,H,W)

Use Cases (UC files):
- UC-001: Addition with scalar broadcasting
- UC-002: Addition with channel-wise bias broadcasting
- UC-003: Addition with spatial map broadcasting
- UC-004: Subtraction with channel-wise bias broadcasting

How to run:
- From the `pytorch` directory with your environment activated:
  - Run all suites: `gt src results`
  - Run this suite only: `gt src/TS-006 results`

Notes:
- All inputs have `requires_grad=True` to enable gradient tracing.
- Each test returns ([inputs], output) as required by gradienttracer.
