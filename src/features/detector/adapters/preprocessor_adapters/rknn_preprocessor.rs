use std::sync::Arc;
use rga::{fill, RgaBuffer, ImageLayout, PixelFormat};
use crate::core::afora_error::AforaError;
use super::cpu::CPUPreprocessor;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::preprocessor::{Preprocessor, PreprocessorConfig};
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;
use crate::stacktrace;

pub struct RknnPreprocessor{
    fallback: PreprocessFallbackPolicy,
    cpu_preprocessor: CPUPreprocessor,
    tensor_spec: Arc<TensorSpec>,
}

impl RknnPreprocessor {

    fn preprocess_with_rknn(&self, frames: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        let input_height = self.tensor_spec.shape[1] as u32;
        let input_width = self.tensor_spec.shape[2] as u32;
        let mut processed_images = Vec::with_capacity(frames.len());

        for frame in &frames {
            let src_layout = ImageLayout::tight(frame.width, frame.height)
                .aligned(PixelFormat::Rgb888);
            let src = stacktrace!("rga_import_src", "preprocessing", RgaBuffer::import_virtual(
            &frame.data,
            src_layout,
            PixelFormat::Rgb888,
        ).map_err(|e| {
            AforaError::PreprocessError(e.to_string())
        }))?;

            //---------------------------------------
            // Crear destino
            //---------------------------------------
            let dst_layout = ImageLayout::tight(input_width, input_height)
                .aligned(PixelFormat::Rgb888);
            let buf_size = (dst_layout.wstride * dst_layout.hstride * 3) as usize;

            // dst_bytes vive en el scope de la iteración, NO dentro del stacktrace!,
            // así que la referencia que guarda `dst` sigue siendo válida después.
            let mut dst_bytes = vec![0u8; buf_size];
            let dst = stacktrace!("rga_dst_setup", "preprocessing", RgaBuffer::import_virtual_mut(
            &mut dst_bytes,
            dst_layout,
            PixelFormat::Rgb888,
        ).map_err(|e| {
            AforaError::PreprocessError(e.to_string())
        }))?;

            //---------------------------------------
            // Letterbox
            //---------------------------------------
            let scale = (input_width as f64 / frame.width as f64)
                .min(input_height as f64 / frame.height as f64);
            let resized_w = (frame.width as f64 * scale).round() as i32;
            let resized_h = (frame.height as f64 * scale).round() as i32;
            let offset_x = (input_width as i32 - resized_w) / 2;
            let offset_y = (input_height as i32 - resized_h) / 2;

            //---------------------------------------
            // Fill
            //---------------------------------------
            stacktrace!("rga_fill", "preprocessing", rga::fill(
            &dst,
            (0, 0, input_width as i32, input_height as i32),
            0x00727272,
        ).map_err(|e| {
            AforaError::PreprocessError(e.to_string())
        }))?;

            //---------------------------------------
            // Resize
            //---------------------------------------
            stacktrace!("rga_resize", "preprocessing", rga::resize_into_rect(
            &src,
            &dst,
            (offset_x, offset_y, resized_w, resized_h),
        ).map_err(|e| {
            AforaError::PreprocessError(e.to_string())
        }))?;

            //---------------------------------------
            // Copia a CPU + conversión a tensor
            //---------------------------------------
            stacktrace!("tensor_convert", "preprocessing", {
            let tight = dst.tight_data();
            drop(dst);
            let tensor = TensorInput::from_rgb(
                &tight,
                self.tensor_spec.as_ref().clone()
            )?;
            processed_images.extend_from_slice(&tensor.data);
            Ok::<_, AforaError>(())
        })?;
        }

        let mut spec = self.tensor_spec.as_ref().clone();
        spec.shape[0] = frames.len() as i64;
        Ok(TensorInput::new(processed_images, spec))
    }

    fn preprocess_with_cpu(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        println!("[Preprocessing]: Fallback to CPU preprocessing");
        self.cpu_preprocessor.preprocess(frame)
    }
}

impl Preprocessor for RknnPreprocessor {
    fn preprocess(&self, frames: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        let result = self.preprocess_with_rknn(frames.clone());
        if result.is_ok() {
            return result;
        }

        let err = result.err().unwrap();

        match &self.fallback {
            PreprocessFallbackPolicy::Cpu => self.preprocess_with_cpu(frames),
            PreprocessFallbackPolicy::Error(msg) => Err(err),
        }
    }

    fn name() -> &'static str {
        "rk3588-preprocessor"
    }

    fn batch_size(&self) -> i64 {
        self.tensor_spec.shape[0]
    }

    fn create(config: PreprocessorConfig) -> Self {
        let cpu_preprocessor = CPUPreprocessor::create(PreprocessorConfig {
            fallback: PreprocessFallbackPolicy::Error("Error de CPU al preprocesar el frame".to_string()),
            target_spec: config.target_spec.clone(),
        });
        Self {
            fallback: config.fallback,
            cpu_preprocessor,
            tensor_spec: config.target_spec,
        }
    }

}