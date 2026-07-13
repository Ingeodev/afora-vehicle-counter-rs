// =============================================================================
// LUT — Lookup Tables para conversión de píxeles sin operaciones aritméticas
// =============================================================================
//
// Elimina divisiones y multiplicaciones del bucle interno de empaquetado.
// Las LUTs se generan en tiempo de compilación (const fn).

/// LUT para normalización u8 -> f32 [0.0, 1.0]
/// Reemplaza: `pixel as f32 / 255.0`
pub static NORM_F32_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
};

/// LUT para cuantización con zero-point 0, scale 1/255
/// Para modelos RKNN que esperan u8 sin normalización, el valor se pasa directo.
/// Esta LUT es identidad pero existe por consistencia de la API.
#[allow(dead_code)]
pub static IDENTITY_U8_LUT: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as u8;
        i += 1;
    }
    lut
};

/// Valor de padding normalizado para F32 (114/255 ≈ 0.447)
pub const PAD_F32: f32 = 114.0 / 255.0;

/// Valor de padding para U8 (gris estándar YOLO)
pub const PAD_U8: u8 = 114;

/// Valor de padding para I8 (signed, offset desde 128)
pub const PAD_I8: i8 = (114i16 - 128) as i8; // -14

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_boundaries() {
        assert!((NORM_F32_LUT[0] - 0.0).abs() < f32::EPSILON);
        assert!((NORM_F32_LUT[255] - 1.0).abs() < f32::EPSILON);
        assert!((NORM_F32_LUT[114] - PAD_F32).abs() < 0.001);
    }

    #[test]
    fn pad_values_consistent() {
        assert_eq!(PAD_U8, 114);
        assert_eq!(PAD_I8, -14);
    }
}
