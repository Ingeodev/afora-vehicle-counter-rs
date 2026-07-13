# Especificación: Motor de Preprocessing Optimizado

## Contexto del Problema

El preprocessing actual representa un bottleneck significativo:

| Métrica | Actual | Objetivo |
|---------|--------|----------|
| Preprocessing | 175-360ms | < 30ms |
| % del ciclo total | ~40-50% | < 10% |

### Causas raíz identificadas

1. **Copia innecesaria en `FrImage::from_vec_u8`** — `.to_vec()` copia ~6MB por frame (1920x1080)
2. **Allocaciones por frame** — `Resizer::new()`, `FrImage::new()` en cada llamada
3. **Doble recorrido del buffer** — `out.fill()` + escritura posterior
4. **Sin reutilización entre batches** — cada batch recrea todo desde cero
5. **Conversión de formato tardía** — siempre se construye `Vec<f32>` y luego se convierte a U8/I8 si es necesario
6. **Layout ignorado** — no se respeta NHWC, siempre se escribe NCHW y se asume que está bien

### Solución encontrada en fast_image_resize v6

La API de `fast_image_resize` v6 expone `ImageRef::new` que acepta `&[u8]` sin copiar:

```rust
use fast_image_resize::images::ImageRef;

// SIN COPIA — usa el slice directamente
let src = ImageRef::new(width, height, &frame.data, PixelType::U8x3)?;
resizer.resize(&src, &mut dst, &options)?;
```

Esto elimina la copia de ~6MB por frame que teníamos con `from_vec_u8(..., rgb_data.to_vec())`.

---

## Arquitectura Propuesta

```
                          ┌─────────────────────────────────────────┐
                          │         PreprocessingEngine             │
                          │  (único, creado al iniciar el Pipeline) │
                          └───────────────┬─────────────────────────┘
                                          │
                                          │ thread_local!
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ScratchContext (por thread)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   Resizer    │  │ ResizeBuffer │  │    Image     │                       │
│  │  (reutiliza) │  │  (destino)   │  │  (destino)   │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura de Archivos

```
src/features/detector/adapters/model_adapters/
├── mod.rs                          # pub mod yolo_model_pipeline; pub mod yolo_onnx_pipeline;
├── yolo_model_pipeline.rs          # Pipeline original (sin modificar)
├── SPECS.md                        # Este documento
├── preprocesing.md                 # Propuesta original (referencia)
└── yolo_onnx_pipeline/
    ├── mod.rs                      # pub mod pipeline; pub mod preprocessing; pub mod scratch; ...
    ├── pipeline.rs                 # YoloOnnxOptimizedPipeline — implementa ModelPipeline
    ├── preprocessing/
    │   ├── mod.rs                  # pub mod engine; pub mod scratch; pub mod lut; pub mod tensor_writer;
    │   ├── engine.rs               # PreprocessingEngine — orquesta el procesamiento
    │   ├── scratch.rs              # ScratchContext — contexto thread-local
    │   ├── lut.rs                  # NORM_LUT — lookup table de normalización
    │   └── tensor_writer.rs        # TensorWriter trait + implementaciones
    └── postprocessing.rs           # Lógica de postprocesamiento (extraída de pipeline)
```

---

## Principio Fundamental: Escritura Directa al Formato Final

**NUNCA** crear un buffer intermedio en un formato para luego convertirlo.

El `TensorSpec` del runtime define:
- `dtype`: F32, U8, I8
- `layout`: NCHW, NHWC
- `shape`: dimensiones esperadas

El preprocessing debe escribir **directamente** en el formato final:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANTI-PATRÓN (código actual)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Construir Vec<f32> en NCHW (siempre)                        │
│  2. Si dtype=U8 → recorrer y cuantizar (pasada extra)           │
│  3. Si layout=NHWC → ??? (no implementado, falla silencioso)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PATRÓN CORRECTO (propuesto)                  │
├─────────────────────────────────────────────────────────────────┤
│  TensorSpec { dtype: F32, layout: Nchw } → escribir f32 en NCHW │
│  TensorSpec { dtype: U8,  layout: Nhwc } → escribir u8 en NHWC  │
│  TensorSpec { dtype: I8,  layout: Nchw } → escribir i8 en NCHW  │
│                                                                 │
│  UNA sola pasada. CERO conversiones posteriores.                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### 1. `TensorWriter` — Núcleo de la Optimización

El TensorWriter es el componente más crítico. Resuelve dtype + layout en tiempo de compilación mediante monomorfización.

```rust
// yolo_onnx_pipeline/preprocessing/tensor_writer.rs

/// Trait sellado que define la escritura especializada al tensor.
/// Cada implementación conoce su dtype y layout en tiempo de compilación.
pub trait TensorWriter: Sized {
    /// Crea el writer sobre un slice de bytes del tamaño correcto.
    fn new(buffer: &mut [u8], side: usize) -> Self;
    
    /// Escribe un pixel RGB en la posición (x, y) del tensor.
    /// La normalización y conversión de layout ocurren aquí.
    fn write_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8);
    
    /// Escribe el valor de padding en la posición (x, y).
    fn write_padding(&mut self, x: usize, y: usize);
    
    /// Valor de padding pre-calculado para este dtype.
    const PAD_VALUE: Self::Element;
    
    type Element;
}

// ═══════════════════════════════════════════════════════════════
// NCHW + F32 (ONNX Runtime típico)
// ═══════════════════════════════════════════════════════════════
pub struct NchwF32Writer<'a> {
    buffer: &'a mut [f32],
    plane_size: usize,  // side * side
}

impl TensorWriter for NchwF32Writer<'_> {
    type Element = f32;
    const PAD_VALUE: f32 = 114.0 / 255.0;  // ~0.447
    
    fn write_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let idx = y * self.side + x;
        // Escritura directa a los 3 planos (R, G, B separados)
        self.buffer[idx] = NORM_LUT[r as usize];
        self.buffer[self.plane_size + idx] = NORM_LUT[g as usize];
        self.buffer[2 * self.plane_size + idx] = NORM_LUT[b as usize];
    }
    
    fn write_padding(&mut self, x: usize, y: usize) {
        let idx = y * self.side + x;
        self.buffer[idx] = Self::PAD_VALUE;
        self.buffer[self.plane_size + idx] = Self::PAD_VALUE;
        self.buffer[2 * self.plane_size + idx] = Self::PAD_VALUE;
    }
}

// ═══════════════════════════════════════════════════════════════
// NHWC + U8 (RKNN típico)
// ═══════════════════════════════════════════════════════════════
pub struct NhwcU8Writer<'a> {
    buffer: &'a mut [u8],
    row_stride: usize,  // side * 3
}

impl TensorWriter for NhwcU8Writer<'_> {
    type Element = u8;
    const PAD_VALUE: u8 = 114;
    
    fn write_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let idx = y * self.row_stride + x * 3;
        // Escritura intercalada (R, G, B juntos por pixel)
        self.buffer[idx] = r;
        self.buffer[idx + 1] = g;
        self.buffer[idx + 2] = b;
    }
    
    fn write_padding(&mut self, x: usize, y: usize) {
        let idx = y * self.row_stride + x * 3;
        self.buffer[idx] = Self::PAD_VALUE;
        self.buffer[idx + 1] = Self::PAD_VALUE;
        self.buffer[idx + 2] = Self::PAD_VALUE;
    }
}
```

**Clave:** El compilador monomorfiza `pack_to_tensor<W: TensorWriter>()`. No hay branches en el bucle interno.

---

### 2. `PreprocessingEngine`

Punto de entrada. Selecciona el TensorWriter correcto según el `TensorSpec` y despacha a la implementación genérica.

```rust
// yolo_onnx_pipeline/preprocessing/engine.rs

pub struct PreprocessingEngine {
    target_side: u32,
}

impl PreprocessingEngine {
    pub fn new(target_side: u32) -> Self;
    
    /// Procesa un batch de frames según el TensorSpec del runtime.
    /// Internamente despacha al TensorWriter correcto.
    pub fn process_batch(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        match (target_spec.dtype, target_spec.layout) {
            (TensorDType::F32, TensorLayout::Nchw) => {
                self.process_batch_impl::<NchwF32Writer>(frames, target_spec)
            }
            (TensorDType::U8, TensorLayout::Nhwc) => {
                self.process_batch_impl::<NhwcU8Writer>(frames, target_spec)
            }
            // ... otras combinaciones
            _ => Err(AforaError::PreprocessError(format!(
                "Combinación dtype={:?} layout={:?} no soportada",
                target_spec.dtype, target_spec.layout
            ))),
        }
    }
    
    /// Implementación genérica que usa el TensorWriter apropiado.
    fn process_batch_impl<W: TensorWriter>(
        &self,
        frames: &[Arc<Frame>],
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        // El buffer se aloca con el tamaño correcto según W::Element
        // Rayon procesa en paralelo, cada thread usa su ScratchContext
        // ...
    }
}
```

---

### 3. `ScratchContext`

Contexto de trabajo por thread. Reutiliza Resizer y buffer de destino.

```rust
// yolo_onnx_pipeline/preprocessing/scratch.rs

pub struct ScratchContext {
    resizer: Resizer,
    dst_image: Image,           // Buffer destino del resize (reutilizado)
    target_side: u32,
    current_size: (u32, u32),   // Tamaño actual del dst_image
}

impl ScratchContext {
    pub fn new(target_side: u32) -> Self;
    
    /// Redimensiona el frame y escribe al tensor usando el TensorWriter.
    pub fn process_frame<W: TensorWriter>(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        writer: &mut W,
    ) -> Result<(), AforaError>;
}
```

---

### 4. LUT de Normalización

```rust
// yolo_onnx_pipeline/preprocessing/lut.rs

/// Lookup table para u8 -> f32 normalizado. Elimina la división del bucle.
pub static NORM_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
};
```

---

## Flujo de Datos Optimizado

```
Frame RGB (HWC, u8)
       │
       │  ← SIN COPIA: ImageRef::new(&frame.data)
       ▼
┌──────────────────┐
│ Resizer::resize  │  ← Reutilizado del ScratchContext
│ (SIMD backend)   │
└────────┬─────────┘
         │
         │  ← Escribe en dst_image (pre-alocado)
         ▼
┌──────────────────────────────────────────────────────────────────┐
│           pack_to_tensor<W: TensorWriter> (una sola pasada)       │
│                                                                   │
│  Para cada (x, y) en la región de la imagen:                      │
│      writer.write_pixel(x + pad_x, y + pad_y, r, g, b)            │
│                                                                   │
│  Para cada (x, y) en la región de padding:                        │
│      writer.write_padding(x, y)                                   │
│                                                                   │
│  El writer ya sabe:                                               │
│      - Si normalizar (F32) o no (U8)                              │
│      - Si escribir planar (NCHW) o intercalado (NHWC)             │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
    TensorInput { data: Vec<u8>, spec: TensorSpec }
```

---

## Instrumentación con stacktrace!

Cada sub-etapa debe medirse para validar mejoras:

```rust
// Dentro de ScratchContext::process_frame

pub fn process_frame(&mut self, frame: &Frame, letterbox: &LetterboxTransform, out: &mut [f32]) -> Result<(), AforaError> {
    // Sub-etapa 1: resize
    stacktrace!("preprocess_resize", "preprocessing", {
        self.resize_frame(frame, letterbox)?
    });
    
    // Sub-etapa 2: pack to tensor
    stacktrace!("preprocess_pack", "preprocessing", {
        self.pack_to_tensor(letterbox, out)
    });
    
    Ok(())
}
```

Tags disponibles para filtrar: `preprocessing`, `into_model`

---

## Fases de Implementación

### Fase 1: Estructura base + TensorWriter + eliminación de copia

**Objetivo:** Establecer la arquitectura correcta desde el inicio.

- [ ] Crear estructura de carpetas `yolo_onnx_pipeline/`
- [ ] Implementar `TensorWriter` trait con `NchwF32Writer`
- [ ] Implementar `NORM_LUT` estática
- [ ] Implementar `ScratchContext` con `ImageRef::new` (sin `.to_vec()`)
- [ ] Implementar `PreprocessingEngine` con thread_local y despacho por TensorSpec
- [ ] Implementar `YoloOnnxOptimizedPipeline` que delega al engine
- [ ] Agregar stacktrace! en sub-etapas: `preprocess_resize`, `preprocess_pack`
- [ ] Medir mejora vs pipeline original

### Fase 2: Reutilización de Resizer y buffers

- [ ] Resizer persistente en ScratchContext
- [ ] dst_image reutilizado entre frames del mismo tamaño escalado
- [ ] Medir mejora

### Fase 3: Eliminación del fill() previo

- [ ] Escribir padding directamente durante pack_to_tensor
- [ ] El TensorWriter ya tiene `write_padding()` para esto
- [ ] Medir mejora

### Fase 4: Soporte NHWC/U8 (RKNN)

- [ ] Implementar `NhwcU8Writer`
- [ ] Agregar combinación en el despacho del engine
- [ ] Tests con TensorSpec de RKNN

---

## Restricciones de Compatibilidad

| Restricción | Solución |
|-------------|----------|
| API pública `ModelPipeline` intacta | Nuevo pipeline implementa el mismo trait |
| Pipeline original sin modificar | Vive en `yolo_model_pipeline.rs`, no se toca |
| Soporte ONNX (f32/NCHW) | Implementación inicial |
| Soporte RKNN (u8/NHWC) | TensorWriter en Fase 5 |
| Thread-safety | thread_local!, sin locks |

---

## Integración con DetectorFactory

Una vez completada la Fase 1, se agregará una nueva variante:

```rust
// detector/mod.rs

pub enum ModelChoice {
    // ... existentes ...
    YoloOnnxOptimized { conf_threshold: f32, input_side: u32, batch_size: u32 },
}

// En build_pipeline:
ModelChoice::YoloOnnxOptimized { conf_threshold, input_side, batch_size } => {
    Box::new(YoloOnnxOptimizedPipeline::new(input_side, conf_threshold, 0.45, batch_size))
}
```

---

## Métricas de Éxito

| Métrica | Actual | Fase 1 | Fase 2 | Fase 4 |
|---------|--------|--------|--------|--------|
| Preprocessing (ms) | 250-360 | < 150 | < 80 | < 30 |
| preprocess_resize (ms) | ? | medido | medido | medido |
| preprocess_pack (ms) | ? | medido | medido | medido |

---

## Referencias

- [fast_image_resize ImageRef](https://docs.rs/fast_image_resize/latest/fast_image_resize/images/struct.ImageRef.html)
- Propuesta original: `preprocesing.md`
- Pipeline original: `yolo_model_pipeline.rs`
