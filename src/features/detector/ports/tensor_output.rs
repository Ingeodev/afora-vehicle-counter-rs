use crate::features::detector::ports::tensor_base::TensorSpec;

/// Salida cruda del runtime. Puede tener múltiples tensores de salida
/// (ej. bboxes + scores por separado, o queries de un DETR).
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub tensors: Vec<(String, Vec<u8>, TensorSpec)>, // (nombre_salida, datos, spec)
}

impl TensorOutput {
    pub fn new(tensors: Vec<(String, Vec<u8>, TensorSpec)>) -> Self {
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Option<&(String, Vec<u8>, TensorSpec)> {
        self.tensors.iter().find(|(n, _, _)| n == name)
    }
}