pub static NORM_F32_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
};

pub const PAD_F32: f32 = 114.0 / 255.0;
pub const PAD_U8: u8 = 114;
pub const PAD_I8: i8 = (114i16 - 128) as i8;

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
