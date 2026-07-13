use fast_image_resize as fr;
use fr::images::Image as FrImage;
use fr::{FilterType as FrFilter, PixelType, ResizeAlg, ResizeOptions, Resizer};
use crate::core::afora_error::AforaError;

const PAD_VALUE: f32 = 114.0 / 255.0;

#[derive(Debug, Clone)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub new_width: u32,
    pub new_height: u32,
    target_side: u32,
}

impl LetterboxTransform {
    pub fn new(original_size: (u32, u32), target_side: u32) -> Self {
        let (ow, oh) = (original_size.0 as f32, original_size.1 as f32);
        let scale = (target_side as f32 / ow).min(target_side as f32 / oh);

        // Redondeamos aquí mismo (antes se redondeaba solo en scaled_width/height,
        // pero pad_x/pad_y se calculaban con el valor sin redondear — quedaba
        // un desfase de sub-píxel entre el padding usado al empacar y el usado
        // en restore_bbox). Ahora todo usa las mismas dimensiones redondeadas.
        let new_width = (ow * scale).round().max(1.0) as u32;
        let new_height = (oh * scale).round().max(1.0) as u32;

        Self {
            scale,
            pad_x: (target_side as f32 - new_width as f32) / 2.0,
            pad_y: (target_side as f32 - new_height as f32) / 2.0,
            new_width,
            new_height,
            target_side,
        }
    }

    /// Redimensiona `rgb_data` (HWC u8, RGB) con SIMD y escribe el resultado
    /// directamente en `out` (CHW f32 normalizado [0,1]), aplicando el
    /// padding en el mismo paso. `out` debe tener tamaño exacto
    /// `3 * target_side * target_side` y corresponder a un solo frame del
    /// batch (el offset dentro del tensor completo lo maneja el llamador).
    ///
    /// Reemplaza el flujo anterior (resize -> canvas 114 -> overlay ->
    /// pasada de normalización aparte) por: resize -> una sola escritura.
    pub fn resize_and_pack_into(
        &self,
        width: u32,
        height: u32,
        rgb_data: &[u8],
        out: &mut [f32],
    ) -> Result<(), AforaError> {
        let side = self.target_side as usize;
        let plane = side * side;

        if out.len() != 3 * plane {
            return Err(AforaError::PreprocessError(format!(
                "Tamaño de buffer de salida incorrecto: esperado {}, recibido {}",
                3 * plane,
                out.len()
            )));
        }

        // 1. Padding normalizado en todo el buffer antes de escribir la imagen real.
        out.fill(PAD_VALUE);

        // 2. Resize SIMD directo del buffer original al tamaño ya escalado
        // (sin pasar por RgbImage ni por un canvas del tamaño completo).
        // NOTA: `to_vec()` es una copia; si tu versión de fast_image_resize
        // expone una variante que acepta un slice prestado (revisa el changelog
        // de tu versión pineada), úsala para eliminar también esta copia.
        let src = FrImage::from_vec_u8(width, height, rgb_data.to_vec(), PixelType::U8x3)
            .map_err(|e| AforaError::PreprocessError(format!("fast_image_resize src: {e}")))?;

        let mut dst = FrImage::new(self.new_width, self.new_height, PixelType::U8x3);

        let mut resizer = Resizer::new();
        resizer
            .resize(
                &src,
                &mut dst,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FrFilter::Bilinear)),
            )
            .map_err(|e| AforaError::PreprocessError(format!("fast_image_resize resize: {e}")))?;

        let scaled = dst.buffer();

        // 3. HWC -> CHW + normalización, escrito directamente en la posición
        // desplazada por el padding. Una sola pasada sobre la región escalada
        // (no sobre el canvas completo, como en la versión con overlay).
        let pad_x = self.pad_x.round() as usize;
        let pad_y = self.pad_y.round() as usize;
        let new_w = self.new_width as usize;
        let new_h = self.new_height as usize;

        for y in 0..new_h {
            let dst_row = (y + pad_y) * side + pad_x;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x;

                out[d] = scaled[s] as f32 / 255.0; // R
                out[plane + d] = scaled[s + 1] as f32 / 255.0; // G
                out[2 * plane + d] = scaled[s + 2] as f32 / 255.0; // B
            }
        }

        Ok(())
    }

    pub fn restore_bbox(&self, cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        (
            (cx - w * 0.5 - self.pad_x) / self.scale,
            (cy - h * 0.5 - self.pad_y) / self.scale,
            (cx + w * 0.5 - self.pad_x) / self.scale,
            (cy + h * 0.5 - self.pad_y) / self.scale,
        )
    }
}