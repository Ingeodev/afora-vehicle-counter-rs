pub(crate) mod cpu;

mod cuda_preprocessor;

#[cfg(feature = "rk3588")]
mod rknn_preprocessor;

#[cfg(feature = "cuda")]
pub use cuda_preprocessor::CudaPreprocessor as PreprocessorImpl;

#[cfg(feature = "rk3588")]
pub use rknn_preprocessor::RknnPreprocessor as PreprocessorImpl;

#[cfg(not(any(feature = "cuda", feature = "rk3588")))]
pub use cpu::CPUPreprocessor as PreprocessorImpl;