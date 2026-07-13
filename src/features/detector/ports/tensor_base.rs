#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    U8,
    I8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}

/// Describe forma y tipo de dato que un runtime espera o produce.
/// El pipeline consulta esto para producir bytes en el formato exacto
/// que el runtime necesita, sin que el pipeline conozca de antemano
/// si el runtime es ONNX (f32/NCHW) o RKNN (u8 cuantizado/NHWC).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub shape: Vec<i64>, // ej. [1, 3, 640, 640]; -1 para dimensiones dinámicas
    pub dtype: TensorDType,
    pub layout: TensorLayout,
}

impl TensorSpec {
    pub fn new(shape: Vec<i64>, dtype: TensorDType, layout: TensorLayout) -> Self {
        Self { shape: shape.to_vec(), dtype, layout }
    }

    /// Verifica si un shape lógico (C, H, W) es compatible con este spec,
    /// contemplando tanto NCHW como NHWC con batch=1.
    pub fn matches_logical_shape(&self, b: u32, c: u32, h: u32, w: u32) -> bool {
        let nchw = [b as i64, c as i64, h as i64, w as i64];
        let nhwc = [b as i64, h as i64, w as i64, c as i64];

        fn matches(spec: &[i64], expected: &[i64]) -> bool {
            spec.len() == expected.len()
                && spec
                .iter()
                .zip(expected.iter())
                .all(|(&s, &e)| s == -1 || s == e)
        }

        matches(&self.shape, &nchw) || matches(&self.shape, &nhwc)
    }
}



