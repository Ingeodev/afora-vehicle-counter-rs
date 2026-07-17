use std::path::PathBuf;
use crate::core::afora_error::AforaError;
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;

/// Abstrae el backend de ejecución (ONNX Runtime, RKNN, TensorRT...).
/// Responsabilidad única: cargar el modelo serializado y ejecutar tensores.
/// NO sabe qué arquitectura de modelo es, ni interpreta la salida semánticamente.
pub trait InferenceRuntime: Send + Sync {
    /// Ejecuta inferencia sobre un tensor ya preprocesado por el ModelPipeline.
    fn run(&mut self, input: &TensorInput) -> Result<TensorOutput, AforaError>;

    /// Spec real de entrada tal como quedó fijado al cargar el modelo.
    fn input_spec(&self) -> &TensorSpec;

    /// Spec real de salida tal como quedó fijado al cargar el modelo.
    fn output_spec(&self) -> &TensorSpec;

    fn runtime_name(&self) -> &'static str;

    /// Warmup opcional — corre N inferencias dummy para estabilizar latencia.
    /// Default no-op; cada runtime puede hacer override si lo necesita
    /// (relevante en RKNN/TensorRT donde el primer forward suele ser más lento).
    fn warmup(&mut self, iterations: u32) -> Result<(), AforaError> {
        let dummy = TensorInput::zeros(self.input_spec());
        for _ in 0..iterations {
            self.run(&dummy)?;
        }
        Ok(())
    }
}

pub struct InferenceRuntimeConfig {
    #[cfg(feature = "cuda")]
    pub model_path: PathBuf,
    #[cfg(feature = "cuda")]
    pub num_threads: usize
}
 