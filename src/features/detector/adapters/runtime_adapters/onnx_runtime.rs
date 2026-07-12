use std::path::PathBuf;
use std::string::String;
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use crate::core::afora_error::AforaError;
use crate::features::detector::application::helpers::{bytes_to_f32, f32_slice_to_bytes};
use crate::features::detector::ports::inference_runtime::InferenceRuntime;
use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;

pub struct OnnxRuntime {
    session: Session,
    input_name: String,
    output_name: String,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
}

impl OnnxRuntime {
    /// Carga el modelo ONNX y captura el spec real de entrada/salida directamente
    /// del grafo (no asumido), para que `Detector::new` pueda validar shapes
    /// contra lo que el `ModelPipeline` espera.
    pub fn load(model_path: PathBuf, num_threads: usize) -> Result<Self, AforaError> {

        let cuda_provider = CUDAExecutionProvider::default();
        let cuda_available = cuda_provider.is_available().unwrap_or(false);

        let session = Session::builder()
            .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?
            .with_execution_providers([
                cuda_provider.build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;

        if cuda_available {
            println!("OnnxRuntime: CUDA execution provider disponible, priorizado.");
        } else {
            println!("OnnxRuntime: CUDA no disponible, usando CPU execution provider.");
        }

        let input_meta = session
            .inputs
            .first()
            .ok_or_else(|| AforaError::RuntimeLoadError("el modelo no declara inputs".into()))?;
        let output_meta = session
            .outputs
            .first()
            .ok_or_else(|| AforaError::RuntimeLoadError("el modelo no declara outputs".into()))?;

        let input_spec = Self::tensor_spec_from_ort_type(&input_meta.input_type)?;
        let output_spec = Self::tensor_spec_from_ort_type(&output_meta.output_type)?;
        let input_name = String::from(input_meta.name.clone());
        let output_name = String::from(output_meta.name.clone());

        Ok(Self {
            session,
            input_name,
            output_name,
            input_spec,
            output_spec,
        })
    }

    /// Convierte el tipo de valor reportado por `ort` a nuestro `TensorSpec` de dominio.
    /// Asume tensores simples f32 en layout NCHW — que es lo que exportan por defecto
    /// tanto PaddleDetection (PP-YOLOE+) como Ultralytics (YOLO11) a ONNX.
    fn tensor_spec_from_ort_type(value_type: &ValueType) -> Result<TensorSpec, AforaError> {
        match value_type {
            ValueType::Tensor { shape, .. } => Ok(TensorSpec::new(
                shape.to_vec(),
                TensorDType::F32,
                TensorLayout::Nchw,
            )),
            _ => Err(AforaError::RuntimeLoadError(
                "solo se soportan modelos con entradas/salidas de tipo tensor simple".into(),
            )),
        }
    }
}




impl InferenceRuntime for OnnxRuntime {
    fn run(&mut self, input: &TensorInput) -> Result<TensorOutput, AforaError> {
        let f32_data = bytes_to_f32(&input.data);
        let tensor = Tensor::from_array((input.spec.shape.clone(), f32_data))
            .map_err(|e| AforaError::InferenceError(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AforaError::InferenceError(e.to_string()))?;

        let output_value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| AforaError::InferenceError(format!(
                "el modelo no devolvió el output esperado '{}'",
                self.output_name
            )))?;

        let (out_shape, out_data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| AforaError::InferenceError(e.to_string()))?;

        Ok(TensorOutput::new(vec![(
            self.output_name.clone(),
            f32_slice_to_bytes(out_data),
            TensorSpec::new(out_shape.to_vec(), TensorDType::F32, TensorLayout::Nchw),
        )]))
    }

    fn input_spec(&self) -> &TensorSpec {
        &self.input_spec
    }

    fn output_spec(&self) -> &TensorSpec {
        &self.output_spec
    }

    fn runtime_name(&self) -> &'static str {
        "onnx"
    }
}