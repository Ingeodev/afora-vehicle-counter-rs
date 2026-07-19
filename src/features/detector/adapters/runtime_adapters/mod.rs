#[cfg(feature = "cuda")]
pub mod onnx_runtime;
#[cfg(feature = "rk3588")]
pub mod rknn_runtime;

#[cfg(feature = "cuda")]
pub use onnx_runtime::OnnxRuntime as RuntimeImpl;

#[cfg(feature = "rk3588")]
pub use rknn_runtime::RknnRuntime as RuntimeImpl;