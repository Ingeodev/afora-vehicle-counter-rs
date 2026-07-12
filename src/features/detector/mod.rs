pub mod ports;
pub mod domain;
pub mod adapters;
pub mod application;

use std::path::PathBuf;
use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::adapters::model_adapters::yolo_model_pipeline::YoloOnnxPipeline;
use crate::features::detector::adapters::runtime_adapters::onnx_runtime::OnnxRuntime;
use crate::features::detector::domain::detection::Detection;
use crate::features::detector::ports::inference_runtime::InferenceRuntime;
use crate::features::detector::ports::model_pipeline::ModelPipeline;
use crate::shared::domain::frame::Frame;
use self::ports::tensor_base::*;


// =============================================================================
// BRIDGE — Detector compone InferenceRuntime + ModelPipeline
// =============================================================================

pub struct Detector {
    runtime: Box<dyn InferenceRuntime>,
    pipeline: Box<dyn ModelPipeline>,
}

impl Detector {
    /// Construye el bridge validando que el shape que el pipeline espera
    /// coincida con el spec real del modelo ya cargado en el runtime.
    pub fn new(
        runtime: Box<dyn InferenceRuntime>,
        pipeline: Box<dyn ModelPipeline>,
    ) -> Result<Self, AforaError> {
        Self::validate_shape(runtime.as_ref(), pipeline.as_ref())?;
        Ok(Self { runtime, pipeline })
    }

    fn validate_shape(
        runtime: &dyn InferenceRuntime,
        pipeline: &dyn ModelPipeline,
    ) -> Result<(), AforaError> {
        let (c, h, w) = pipeline.expected_input_shape();
        let spec = runtime.input_spec();
        if !spec.matches_logical_shape(c, h, w) {
            return Err(AforaError::ShapeMismatch {
                expected: (c, h, w),
                actual: spec.shape.clone(),
            });
        }
        Ok(())
    }

    /// Ejecuta el pipeline completo: preprocess -> run -> postprocess.
    pub fn detect(&mut self, frame: Arc<Frame>) -> Result<Vec<Detection>, AforaError> {
        let input = self.pipeline.preprocess(frame.clone(), self.runtime.input_spec())?;
        let output = self.runtime.run(&input)?;
        self.pipeline.postprocess(output, frame.original_size())
    }

    pub fn runtime_name(&self) -> &'static str {
        self.runtime.runtime_name()
    }

    pub fn model_name(&self) -> &'static str {
        self.pipeline.model_name()
    }

    /// Corre inferencias dummy para estabilizar latencia (delega al runtime).
    pub fn warmup(&mut self, iterations: u32) -> Result<(), AforaError> {
        self.runtime.warmup(iterations)
    }
}

// =============================================================================
// FACTORY — ensambla runtime + pipeline según configuración
// =============================================================================

/// Selección de runtime a construir. Cada variante lleva los parámetros
/// mínimos necesarios para cargar ese backend específico.
pub enum RuntimeChoice {
    Onnx {
        model_path: PathBuf,
        num_threads: usize,
    }
}

/// Selección de arquitectura de modelo a construir.
pub enum ModelChoice {
    PpYoloePlusS { conf_threshold: f32 },
    YoloOnnx { conf_threshold: f32 },
    RfDetr { conf_threshold: f32 },
}

pub struct DetectorFactory;

impl DetectorFactory {
    /// Punto único de ensamblaje. Las implementaciones concretas de
    /// InferenceRuntime y ModelPipeline referenciadas aquí (OnnxRuntime,
    /// RknnRuntime, PpYoloePipeline, etc.) viven en `infrastructure/` y
    /// deben implementarse por separado; este factory solo orquesta.
    pub fn build(
        runtime_choice: RuntimeChoice,
        model_choice: ModelChoice,
    ) -> Result<Detector, AforaError> {
        let runtime: Box<dyn InferenceRuntime> = Self::build_runtime(runtime_choice)?;
        let pipeline: Box<dyn ModelPipeline> = Self::build_pipeline(model_choice);
        Detector::new(runtime, pipeline)
    }

    fn build_runtime(choice: RuntimeChoice) -> Result<Box<dyn InferenceRuntime>, AforaError> {
        match choice {
            RuntimeChoice::Onnx { model_path, num_threads } => {
                // Reemplazar por: OnnxRuntime::load(&model_path, num_threads)
                Ok(Box::new(OnnxRuntime::load(model_path, num_threads)?))
            }
        }
    }

    fn build_pipeline(choice: ModelChoice) -> Box<dyn ModelPipeline> {
        match choice {
            ModelChoice::PpYoloePlusS { conf_threshold } => {
                let _ = conf_threshold;
                // Reemplazar por: Box::new(PpYoloePipeline::new(conf_threshold))
                unimplemented!("conectar con infrastructure::pipelines::PpYoloePipeline::new")
            }
            ModelChoice::YoloOnnx { conf_threshold } => {
                let _ = conf_threshold;
                Box::new(YoloOnnxPipeline::new(640, conf_threshold, 0.45))
            }
            ModelChoice::RfDetr { conf_threshold } => {
                let _ = conf_threshold;
                unimplemented!("conectar con infrastructure::pipelines::RfDetrPipeline::new")
            }
        }
    }
}

// =============================================================================
// TESTS DE CONTRATO — validan reglas del bridge sin depender de infra concreta
// =============================================================================

#[cfg(test)]
mod tests {
    use ort::value::Shape;
    use crate::features::detector::ports::tensor_input::TensorInput;
    use crate::features::detector::ports::tensor_output::TensorOutput;
    use super::*;

    /// Runtime de prueba: acepta cualquier capacidad, útil para validar
    /// solo la lógica de shape-matching del Detector.
    struct FakeRuntime {
        input_spec: TensorSpec,
        output_spec: TensorSpec,
    }

    impl InferenceRuntime for FakeRuntime {
        fn run(&mut self, _input: &TensorInput) -> Result<TensorOutput, AforaError> {
            Ok(TensorOutput::new(vec![]))
        }
        fn input_spec(&self) -> &TensorSpec { &self.input_spec }
        fn output_spec(&self) -> &TensorSpec { &self.output_spec }
        fn runtime_name(&self) -> &'static str { "fake_runtime" }
    }

    struct FakePipeline {
        expected_shape: (u32, u32, u32),
    }

    impl ModelPipeline for FakePipeline {
        fn preprocess(&self, _frame: Arc<Frame>, _target_spec: &TensorSpec) -> Result<TensorInput, AforaError> {
            Ok(TensorInput::new(vec![], self.expected_shape_as_spec()))
        }
        fn postprocess(&self, _output: TensorOutput, _original_size: (u32, u32)) -> Result<Vec<Detection>, AforaError> {
            Ok(vec![])
        }
        fn model_name(&self) -> &'static str { "fake_model" }
        fn expected_input_shape(&self) -> (u32, u32, u32) { self.expected_shape }
        //fn class_taxonomy(&self) -> &'static [&'static str] { &["car", "motorcycle"] }
    }

    impl FakePipeline {
        fn expected_shape_as_spec(&self) -> TensorSpec {
            let (c, h, w) = self.expected_shape;
            TensorSpec::new(vec![1, c as i64 , h as i64 , w as i64], TensorDType::F32, TensorLayout::Nchw)
        }
    }

    #[test]
    fn detector_construye_ok_cuando_shape_coincide() {
        let runtime = Box::new(FakeRuntime {
            input_spec: TensorSpec::new(vec![1, 3, 640, 640], TensorDType::F32, TensorLayout::Nchw),
            output_spec: TensorSpec::new(vec![1, 100, 6], TensorDType::F32, TensorLayout::Nchw),
        });
        let pipeline = Box::new(FakePipeline {
            expected_shape: (3, 640, 640),
        });

        let detector = Detector::new(runtime, pipeline);
        assert!(detector.is_ok());
    }

    #[test]
    fn detector_falla_por_shape_mismatch() {
        let runtime = Box::new(FakeRuntime {
            input_spec: TensorSpec::new(vec![1, 3, 320, 320], TensorDType::F32, TensorLayout::Nchw),
            output_spec: TensorSpec::new(vec![1, 100, 6], TensorDType::F32, TensorLayout::Nchw),
        });
        let pipeline = Box::new(FakePipeline {
            expected_shape: (3, 640, 640), // no coincide con el runtime (320x320)
        });

        let result = Detector::new(runtime, pipeline);
        assert!(matches!(result, Err(AforaError::ShapeMismatch { .. })));
    }
}