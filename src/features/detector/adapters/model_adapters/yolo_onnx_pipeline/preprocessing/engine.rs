// =============================================================================
// PREPROCESSING ENGINE — Orquestador del preprocessing optimizado
// =============================================================================
//
// Punto de entrada principal para el preprocessing. Responsabilidades:
// 1. Despachar al formato correcto según TensorSpec
// 2. Coordinar el procesamiento paralelo con Rayon
// 3. Gestionar thread_local para ScratchContext

use std::cell::RefCell;
use std::sync::Arc;

use rayon::prelude::*;

use crate::core::afora_error::AforaError;
use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;
use crate::shared::utilities::letterbox_transform::LetterboxTransform;

use crate::stacktrace;

use super::scratch::ScratchContext;

// =============================================================================
// THREAD LOCAL STORAGE
// =============================================================================

thread_local! {
    static SCRATCH: RefCell<Option<ScratchContext>> = const { RefCell::new(None) };
}

fn with_scratch<F, R>(target_side: u32, f: F) -> R
where
    F: FnOnce(&mut ScratchContext) -> R,
{
    SCRATCH.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let ctx = borrow.get_or_insert_with(|| ScratchContext::new(target_side));

        if ctx.target_side != target_side {
            *ctx = ScratchContext::new(target_side);
        }

        f(ctx)
    })
}

// =============================================================================
// ENGINE
// =============================================================================

#[derive(Clone)]
pub struct PreprocessingEngine {
    target_side: u32,
}

impl PreprocessingEngine {
    pub fn new(target_side: u32) -> Self {
        Self { target_side }
    }

    pub fn process_batch(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        if frames.is_empty() {
            return Err(AforaError::PreprocessError("Batch vacío".into()));
        }

        match (target_spec.dtype, target_spec.layout) {
            (TensorDType::F32, TensorLayout::Nchw) => {
                self.process_batch_f32_nchw(frames, target_spec)
            }
            (TensorDType::U8, TensorLayout::Nhwc) => {
                self.process_batch_u8_nhwc(frames, target_spec)
            }
            (TensorDType::U8, TensorLayout::Nchw) => {
                self.process_batch_u8_nchw(frames, target_spec)
            }
            (TensorDType::I8, TensorLayout::Nchw) => {
                self.process_batch_i8_nchw(frames, target_spec)
            }
            (TensorDType::I8, TensorLayout::Nhwc) => {
                self.process_batch_i8_nhwc(frames, target_spec)
            }
            (TensorDType::F32, TensorLayout::Nhwc) => Err(AforaError::PreprocessError(
                "F32 + NHWC no soportado".into(),
            )),
        }
    }

    // =========================================================================
    // IMPLEMENTACIONES ESPECIALIZADAS
    // =========================================================================

    fn process_batch_f32_nchw(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        let side = self.target_side as usize;
        let batch_size = frames.len();
        let elements_per_frame = 3 * side * side;

        // Medir allocación del buffer
        let mut buffer = stacktrace!("alloc_output_buffer", "preprocessing", {
            vec![0.0f32; batch_size * elements_per_frame]
        });

        buffer
            .par_chunks_mut(elements_per_frame)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| {
                // Medir creación de letterbox
                let letterbox = stacktrace!("create_letterbox", "preprocessing", {
                    LetterboxTransform::new((frame.width, frame.height), self.target_side)
                });
                
                // Medir acceso a thread_local + procesamiento
                stacktrace!("thread_local_process", "preprocessing", {
                    with_scratch(self.target_side, |ctx| {
                        ctx.process_frame_nchw_f32(frame, &letterbox, chunk)
                    })
                })
            })?;

        // Medir conversión a bytes
        let bytes = stacktrace!("convert_to_bytes", "preprocessing", {
            f32_slice_to_bytes(&buffer)
        });
        
        let spec = self.build_output_spec_nchw(batch_size, target_spec);

        Ok(TensorInput::new(bytes, spec))
    }

    fn process_batch_u8_nhwc(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        let side = self.target_side as usize;
        let batch_size = frames.len();
        let bytes_per_frame = side * side * 3;

        let mut buffer = vec![0u8; batch_size * bytes_per_frame];

        buffer
            .par_chunks_mut(bytes_per_frame)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| {
                let letterbox = LetterboxTransform::new((frame.width, frame.height), self.target_side);
                with_scratch(self.target_side, |ctx| {
                    ctx.process_frame_nhwc_u8(frame, &letterbox, chunk)
                })
            })?;

        let spec = self.build_output_spec_nhwc(batch_size, target_spec);

        Ok(TensorInput::new(buffer, spec))
    }

    fn process_batch_u8_nchw(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        let side = self.target_side as usize;
        let batch_size = frames.len();
        let bytes_per_frame = 3 * side * side;

        let mut buffer = vec![0u8; batch_size * bytes_per_frame];

        buffer
            .par_chunks_mut(bytes_per_frame)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| {
                let letterbox = LetterboxTransform::new((frame.width, frame.height), self.target_side);
                with_scratch(self.target_side, |ctx| {
                    ctx.process_frame_nchw_u8(frame, &letterbox, chunk)
                })
            })?;

        let spec = self.build_output_spec_nchw(batch_size, target_spec);

        Ok(TensorInput::new(buffer, spec))
    }

    fn process_batch_i8_nchw(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        let side = self.target_side as usize;
        let batch_size = frames.len();
        let bytes_per_frame = 3 * side * side;

        let mut buffer = vec![0i8; batch_size * bytes_per_frame];

        buffer
            .par_chunks_mut(bytes_per_frame)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| {
                let letterbox = LetterboxTransform::new((frame.width, frame.height), self.target_side);
                with_scratch(self.target_side, |ctx| {
                    ctx.process_frame_nchw_i8(frame, &letterbox, chunk)
                })
            })?;

        let bytes = i8_slice_to_bytes(&buffer);
        let spec = self.build_output_spec_nchw(batch_size, target_spec);

        Ok(TensorInput::new(bytes, spec))
    }

    fn process_batch_i8_nhwc(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        let side = self.target_side as usize;
        let batch_size = frames.len();
        let bytes_per_frame = side * side * 3;

        let mut buffer = vec![0i8; batch_size * bytes_per_frame];

        buffer
            .par_chunks_mut(bytes_per_frame)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| {
                let letterbox = LetterboxTransform::new((frame.width, frame.height), self.target_side);
                with_scratch(self.target_side, |ctx| {
                    ctx.process_frame_nhwc_i8(frame, &letterbox, chunk)
                })
            })?;

        let bytes = i8_slice_to_bytes(&buffer);
        let spec = self.build_output_spec_nhwc(batch_size, target_spec);

        Ok(TensorInput::new(bytes, spec))
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    #[inline]
    fn build_output_spec_nchw(&self, batch_size: usize, base_spec: &TensorSpec) -> TensorSpec {
        let side = self.target_side as i64;
        let mut spec = base_spec.clone();
        spec.shape = vec![batch_size as i64, 3, side, side];
        spec
    }

    #[inline]
    fn build_output_spec_nhwc(&self, batch_size: usize, base_spec: &TensorSpec) -> TensorSpec {
        let side = self.target_side as i64;
        let mut spec = base_spec.clone();
        spec.shape = vec![batch_size as i64, side, side, 3];
        spec
    }
}

// =============================================================================
// CONVERSIÓN DE BYTES
// =============================================================================

#[inline]
fn f32_slice_to_bytes(slice: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(slice.len() * 4);
    for &val in slice {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

#[inline]
fn i8_slice_to_bytes(slice: &[i8]) -> Vec<u8> {
    slice.iter().map(|&v| v as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, color: (u8, u8, u8)) -> Arc<Frame> {
        let (r, g, b) = color;
        let data: Vec<u8> = (0..(width * height)).flat_map(|_| [r, g, b]).collect();

        Arc::new(Frame {
            data,
            width,
            height,
        })
    }

    #[test]
    fn engine_processes_single_frame_f32_nchw() {
        let engine = PreprocessingEngine::new(640);
        let frame = create_test_frame(1920, 1080, (128, 64, 32));

        let spec = TensorSpec::new(
            vec![1, 3, 640, 640],
            TensorDType::F32,
            TensorLayout::Nchw,
        );

        let result = engine.process_batch(&[frame], &spec).unwrap();

        assert_eq!(result.spec.shape[0], 1);
        assert_eq!(result.data.len(), 1 * 3 * 640 * 640 * 4);
    }

    #[test]
    fn engine_processes_batch_u8_nhwc() {
        let engine = PreprocessingEngine::new(320);
        let frames: Vec<Arc<Frame>> = (0..4)
            .map(|i| create_test_frame(640, 480, (i * 50, i * 30, i * 20)))
            .collect();

        let spec = TensorSpec::new(
            vec![4, 320, 320, 3],
            TensorDType::U8,
            TensorLayout::Nhwc,
        );

        let result = engine.process_batch(&frames, &spec).unwrap();

        assert_eq!(result.spec.shape[0], 4);
        assert_eq!(result.data.len(), 4 * 320 * 320 * 3);
    }

    #[test]
    fn engine_rejects_empty_batch() {
        let engine = PreprocessingEngine::new(640);
        let spec = TensorSpec::new(
            vec![1, 3, 640, 640],
            TensorDType::F32,
            TensorLayout::Nchw,
        );

        let result = engine.process_batch(&[], &spec);
        assert!(result.is_err());
    }

    #[test]
    fn engine_handles_different_frame_sizes() {
        let engine = PreprocessingEngine::new(640);

        let frames = vec![
            create_test_frame(1920, 1080, (255, 0, 0)),
            create_test_frame(1080, 1920, (0, 255, 0)),
            create_test_frame(640, 640, (0, 0, 255)),
        ];

        let spec = TensorSpec::new(
            vec![3, 3, 640, 640],
            TensorDType::F32,
            TensorLayout::Nchw,
        );

        let result = engine.process_batch(&frames, &spec).unwrap();

        assert_eq!(result.spec.shape[0], 3);
    }
}
