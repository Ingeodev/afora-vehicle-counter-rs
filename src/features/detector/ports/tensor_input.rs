use crate::core::afora_error::AforaError;
use crate::features::detector::adapters::preprocessor_adapters::cpu::lut::NORM_F32_LUT;
use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout, TensorSpec};

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

    pub fn from_rgb(
        rgb: &[u8],
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {
        let width = spec.shape[1] as u32;
        let height = spec.shape[2] as u32;
        match (spec.dtype, spec.layout) {
            (TensorDType::U8, TensorLayout::Nhwc) => {
                Self::from_rgb_u8_nhwc(rgb, width, height, spec)
            }

            (TensorDType::U8, TensorLayout::Nchw) => {
                Self::from_rgb_u8_nchw(rgb, width, height, spec)
            }

            (TensorDType::I8, TensorLayout::Nhwc) => {
                Self::from_rgb_i8_nhwc(rgb, width, height, spec)
            }

            (TensorDType::I8, TensorLayout::Nchw) => {
                Self::from_rgb_i8_nchw(rgb, width, height, spec)
            }

            (TensorDType::F32, TensorLayout::Nchw) => {
                Self::from_rgb_f32_nchw(rgb, width, height, spec)
            }

            (TensorDType::F32, TensorLayout::Nhwc) => {
                Err(AforaError::PreprocessError(
                    "F32 NHWC no soportado".into(),
                ))
            }
        }
    }

    fn from_rgb_u8_nhwc(
        rgb: &[u8],
        width: u32,
        height: u32,
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {

        debug_assert_eq!(rgb.len(), (width * height * 3) as usize);

        Ok(Self {
            data: rgb.to_vec(),
            spec,
        })
    }

    fn from_rgb_u8_nchw(
        rgb: &[u8],
        width: u32,
        height: u32,
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {

        let plane = (width * height) as usize;

        let mut out = vec![0u8; plane * 3];

        for i in 0..plane {
            let s = i * 3;
            out[i] = rgb[s];
            out[plane + i] = rgb[s + 1];
            out[plane * 2 + i] = rgb[s + 2];
        }

        Ok(Self {
            data: out,
            spec,
        })
    }

    fn from_rgb_i8_nhwc(
        rgb: &[u8],
        width: u32,
        height: u32,
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {

        debug_assert_eq!(rgb.len(), (width * height * 3) as usize);

        let mut out = Vec::with_capacity(rgb.len());

        for &v in rgb {
            out.push((v as i16 - 128) as u8);
        }

        Ok(Self {
            data: out,
            spec,
        })
    }

    fn from_rgb_i8_nchw(
        rgb: &[u8],
        width: u32,
        height: u32,
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {

        let plane = (width * height) as usize;

        let mut out = vec![0u8; plane * 3];

        for i in 0..plane {
            let s = i * 3;

            out[i] = (rgb[s] as i16 - 128) as u8;
            out[plane + i] = (rgb[s + 1] as i16 - 128) as u8;
            out[plane * 2 + i] = (rgb[s + 2] as i16 - 128) as u8;
        }

        Ok(Self {
            data: out,
            spec,
        })
    }

    fn from_rgb_f32_nchw(
        rgb: &[u8],
        width: u32,
        height: u32,
        spec: TensorSpec,
    ) -> Result<Self, AforaError> {

        let plane = (width * height) as usize;

        let mut out = Vec::with_capacity(plane * 3 * 4);

        let mut r = vec![0f32; plane];
        let mut g = vec![0f32; plane];
        let mut b = vec![0f32; plane];

        for i in 0..plane {
            let s = i * 3;

            r[i] = NORM_F32_LUT[rgb[s] as usize];
            g[i] = NORM_F32_LUT[rgb[s + 1] as usize];
            b[i] = NORM_F32_LUT[rgb[s + 2] as usize];
        }

        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }

        for v in g {
            out.extend_from_slice(&v.to_le_bytes());
        }

        for v in b {
            out.extend_from_slice(&v.to_le_bytes());
        }

        Ok(Self {
            data: out,
            spec,
        })
    }
}