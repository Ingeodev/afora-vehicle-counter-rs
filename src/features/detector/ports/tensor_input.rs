use crate::features::detector::ports::tensor_base::{TensorDType, TensorSpec};

/// Tensor de entrada ya preprocesado por el ModelPipeline, listo para
/// que el InferenceRuntime lo ejecute. Los datos viajan como bytes crudos
/// más su spec, para no forzar un tipo único (f32) entre runtimes distintos.
#[derive(Debug, Clone)]
pub struct TensorInput {
    pub data: Vec<u8>,
    pub spec: TensorSpec,
}

impl TensorInput {
    pub fn new(data: Vec<u8>, spec: TensorSpec) -> Self {
        Self { data, spec }
    }

    /// Tensor de ceros del tamaño esperado por un spec — útil para warmup.
    pub fn zeros(spec: &TensorSpec) -> Self {
        let element_count: i64 = spec.shape.iter().map(|d| d.max(&1)).product();
        let byte_size = match spec.dtype {
            TensorDType::F32 => element_count as usize * 4,
            TensorDType::U8 | TensorDType::I8 => element_count as usize,
        };
        Self { data: vec![0u8; byte_size], spec: spec.clone() }
    }
}