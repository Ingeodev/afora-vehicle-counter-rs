use std::option::Option::None;
use std::fs;

use rknpu2::{
    RKNN,
    api::{RknnInitFlags, Priority},
    io::{
        buffer::{BufMutView, BufView},
        input::Input,
        output::{Output, OutputKind},
    },
    query::{InputAttr, InputOutputNum, OutputAttr, TensorAttrView},
    tensor::{QuantTypeKind, TensorFormat, TensorFormatKind},
};

use rknpu2::api::linked::LinkedAPI;

use crate::core::afora_error::AforaError;
use crate::features::detector::application::helpers::{bytes_to_f32, f32_slice_to_bytes};
use crate::features::detector::ports::inference_runtime::{InferenceRuntime, InferenceRuntimeConfig};
use crate::features::detector::ports::tensor_base::{QuantParams, TensorDType, TensorLayout, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::stacktrace;

pub struct RknnRuntime {
    //Se selecciona el tipo LinkedAPI porque librknnrt.so se resuelve en tiempo de compilacion.
    //Si se resolviera en tiempo de ejecicion, entonces tocaría usar RuntimeApi, pero solo se usa
    //si la libreria se piensa compilar en otra maquina que no cuenta con esa librería resuelta, que no es nuestro caso.
    model: RKNN<LinkedAPI>,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
    /// n_elems del output, para preasignar el buffer de destino sin volver a consultar en cada `run`.
    output_len: usize,
}

impl RknnRuntime {
        pub fn load(config: InferenceRuntimeConfig) -> Result<Self, AforaError> {
            let mut model_data = fs::read(&config.model_path)
                .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;


            //Si en algun momento pienso correr inferencia en diferentes procesos con el mismo modelo
            //entonces podría usar uno de estos flags para compartir los pesos que se encuentran cargados
            //y asi evitar cargarlos multiples veces.
            let flags = RknnInitFlags::empty().with_priority(Priority::High);

            let model = RKNN::new(&mut model_data, flags)
                .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;

            let io_num: InputOutputNum = model
                .query()
                .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;

            if io_num.input_num() == 0 || io_num.output_num() == 0 {
                return Err(AforaError::RuntimeLoadError(
                    "el modelo RKNN no declara inputs/outputs".into(),
                ));
            }

            let input_attr: InputAttr = model
                .query_with_input(0u32)
                .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;

            let output_attr: OutputAttr = model
                .query_with_input(0u32)
                .map_err(|e| AforaError::RuntimeLoadError(e.to_string()))?;

            let input_is_quantized = !matches!(input_attr.qnt_type(), QuantTypeKind::None(_));


            let input_spec = {
                let shape = input_attr.dims().iter().map(|&d| d as i64).collect();
                let dtype = if input_is_quantized { TensorDType::I8 } else { TensorDType::F32 };
                let spec = TensorSpec::new(shape, dtype, TensorLayout::Nhwc);
                if input_is_quantized {
                    spec.with_quant(input_attr.scale(), input_attr.zero_point())
                } else {
                    spec
                }
            };

            let output_spec = TensorSpec::new(
                output_attr.dims().iter().map(|&d| d as i64).collect(),
                TensorDType::F32,
                TensorLayout::Nchw,
                );


            Ok(Self {
                model,
                input_spec,
                output_spec,
                output_len: output_attr.num_elements() as usize,
            })
        }

}

impl InferenceRuntime for RknnRuntime {
    fn run(&mut self, input: &TensorInput) -> Result<TensorOutput, AforaError> {
        debug_assert_eq!(
            &input.spec, &self.input_spec,
            "el preprocessor RKNN debe entregar bytes ya en el dtype/quant declarado por self.input_spec"
        );

        // Fase 1: prep en CPU (conversión de dtype + construcción del Input) — no debería
        // ser representativa, si aparece pesada aquí es una señal de conversión de datos costosa
        // Solo se allocatea si el dtype es F32; para I8 esto queda en None sin costo extra
        let f32_buf: Option<Vec<f32>> = match self.input_spec.dtype {
            TensorDType::F32 => Some(bytes_to_f32(&input.data)),
            _ => None,
        };

        let rknn_input = stacktrace!("input_prep", "runtime_inference", {
        let buf_view = match self.input_spec.dtype {
            TensorDType::I8 => BufView::I8(as_i8_slice(&input.data)),
            TensorDType::F32 => BufView::F32(f32_buf.as_ref().unwrap()),
            TensorDType::U8 => unreachable!(
                "RknnRuntime::load nunca produce input_spec con dtype U8"
            ),
        };
        Input::new(
            0,
            buf_view,
            true, // pass_through: el preprocessor ya entregó exactamente lo que el modelo espera
            TensorFormatKind::NHWC(TensorFormat::NHWC),
        )
    });

        // Fase 2: transferencia host -> NPU (copia/DMA hacia buffers del modelo)
        stacktrace!("rknn_set_input", "runtime_inference", self.model
    .set_inputs(rknn_input)
    .map_err(|e| AforaError::InferenceError(e.to_string())))?;

        // Fase 3: ejecución real en el NPU — la sospechosa principal dado el delta que mediste
        stacktrace!("rknn_run", "runtime_inference", self.model
    .run()
    .map_err(|e| AforaError::InferenceError(e.to_string())))?;

        let mut logits = vec![0.0f32; self.output_len];
        let output = Output {
            index: 0,
            kind: OutputKind::Preallocated {
                buf: BufMutView::F32(&mut logits),
                want_float: true, // el runtime dequantiza usando el scale/zp internos del output
            },
        };

        // Fase 4: transferencia NPU -> host + dequantización (want_float=true dispara el escalado
        // scale/zp; si esta fase pesa, la dequantización en el driver puede ser el segundo sospechoso)
        stacktrace!("rknn_get_outputs", "runtime_inference", self.model
    .get_outputs(&mut vec![output])
    .map_err(|e| AforaError::InferenceError(e.to_string())))?;

        Ok(TensorOutput::new(vec![(
            "output0".to_string(),
            f32_slice_to_bytes(&logits),
            self.output_spec.clone(),
        )]))
    }

    fn input_spec(&self) -> &TensorSpec {
        &self.input_spec
    }

    fn output_spec(&self) -> &TensorSpec {
        &self.output_spec
    }

    fn runtime_name(&self) -> &'static str {
        "rknn"
    }
}

// i8 y u8 tienen mismo tamaño/alineación que los bytes crudos; reinterpretar es seguro,
// no es un transmute arriesgado — el preprocessor ya garantizó que input.data son i8 válidos.
fn as_i8_slice(bytes: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) }
}