---

# Prompt: Implementación de un Motor de Preprocessing Ultra Optimizado para Inferencia

## Objetivo

Rediseñar completamente el módulo de preprocessing de la librería de inferencia.

**El objetivo NO es mejorar el código existente.**

El objetivo es construir un preprocessing cuyo tiempo sea una fracción del actual, acercándose al límite impuesto por el propio algoritmo de resize.

Debe ser un motor de preprocessing comparable en filosofía a TensorRT, OpenVINO, NCNN u OpenCV DNN.

La API pública (`ModelPipeline`) debe mantenerse compatible.

---

# Objetivos de rendimiento

Priorizar, en este orden:

1. Reducir al mínimo absoluto el movimiento de memoria.
2. Eliminar asignaciones dinámicas repetidas.
3. Reducir el número de recorridos sobre los píxeles.
4. Maximizar localidad de caché.
5. Aprovechar SIMD.
6. Paralelizar únicamente donde realmente produzca beneficios.
7. Mantener compatibilidad con:

    * x86_64
    * ARM64 (RK3588)
    * CUDA (preprocesamiento en CPU)
    * ONNX Runtime
    * RKNN
    * futuros runtimes

No optimizar para una plataforma específica.

---

# Arquitectura objetivo

```text
Frame Batch
      │
      ▼
Rayon Worker Pool
      │
      ▼
Thread Local Context
      │
      ▼
Resize Backend
      │
      ▼
Tensor Writer
      │
      ▼
TensorInput
```

Cada worker procesa una imagen completa.

Nunca dividir una imagen entre múltiples hilos.

---

# Fase 1 — Eliminar asignaciones dinámicas

## Objetivo

Durante el procesamiento normal NO debe ejecutarse:

```
Vec::new()

Vec::with_capacity()

resize()

reserve()

FrImage::new()

Resizer::new()
```

Todo debe existir previamente.

---

## Implementación

Crear:

```
ScratchContext
```

que contenga:

```
Resizer

Resize Buffer

Tensor Buffer

LUT

Buffers temporales
```

Cada hilo posee su propio contexto.

Nunca compartirlo.

---

# Fase 2 — Thread Local Storage

Usar:

```rust
thread_local!
```

para almacenar:

```
ScratchContext
```

Cada hilo reutiliza siempre:

* mismo Resizer
* mismo Resize Buffer
* misma LUT
* mismos buffers auxiliares

Eliminar completamente sincronización.

No usar:

```
Mutex

RwLock

Arc<Mutex<_>>
```

---

# Fase 3 — Memory Pool

Crear un pool permanente de:

```
Tensor Buffers
```

alineados.

Nunca asignar memoria por batch.

El tensor debe solicitarse al pool y devolverse al finalizar.

---

# Fase 4 — Memoria alineada

Todos los buffers grandes deben alinearse a:

```
64 bytes
```

para favorecer:

* AVX2
* AVX512
* NEON

Evitar buffers sin alineación explícita.

---

# Fase 5 — LUT para normalización

Eliminar completamente:

```rust
pixel as f32 / 255.0
```

Crear:

```
static LUT[256]
```

donde:

```
LUT[i]
```

sea el valor normalizado.

Durante el procesamiento únicamente hacer:

```
out = LUT[pixel]
```

---

# Fase 6 — TensorWriter especializado

Eliminar completamente:

```
if dtype == ...

if layout == ...
```

del bucle de procesamiento.

Crear implementaciones especializadas:

```
TensorWriter<F32,NCHW>

TensorWriter<F32,NHWC>

TensorWriter<U8,NCHW>

TensorWriter<U8,NHWC>

TensorWriter<I8,NCHW>

TensorWriter<I8,NHWC>
```

Resolver mediante genéricos.

El compilador debe monomorfizar.

Cero ramas dentro del bucle.

---

# Fase 7 — Escritura directa

Eliminar índices repetitivos:

Actualmente:

```
out[d]

out[plane+d]

out[2*plane+d]
```

Reemplazar por punteros:

```
r_ptr

g_ptr

b_ptr
```

Incrementarlos.

Evitar multiplicaciones dentro del bucle.

---

# Fase 8 — Loop Unrolling

Desenrollar manualmente el bucle.

Procesar:

```
8

16

32
```

pixeles por iteración.

Reducir overhead de saltos.

---

# Fase 9 — SIMD

Implementar usando:

```
std::simd
```

cuando sea estable para el objetivo del proyecto.

Si se requiere mayor rendimiento:

usar implementaciones específicas mediante:

```
#[target_feature]
```

para:

```
AVX2

AVX512

NEON
```

Mantener fallback escalar.

---

# Fase 10 — Prefetch

Añadir prefetch cuando la arquitectura lo permita.

Mientras se procesan los píxeles actuales:

prefetch de las siguientes líneas.

Reducir cache misses.

---

# Fase 11 — Non Temporal Stores

Cuando el tensor vaya directamente al runtime:

usar almacenamiento no temporal cuando esté disponible.

Evitar contaminar caché con un buffer que no volverá a leerse por CPU.

---

# Fase 12 — Reducir recorridos

El procesamiento debe realizar únicamente:

```
Resize

↓

Tensor Final
```

Evitar:

```
Resize

↓

Imagen temporal

↓

Normalización

↓

Tensor
```

Cada recorrido adicional de memoria debe justificarse.

---

# Fase 13 — Padding fusionado

Nunca construir un canvas completo.

El padding debe escribirse directamente en el tensor.

Mientras se copian los píxeles escalados.

---

# Fase 14 — Conversión de layout

Nunca crear buffers intermedios.

La conversión:

```
HWC

↓

NCHW
```

debe hacerse durante la escritura.

No posteriormente.

---

# Fase 15 — Conversión de tipo

Nunca generar primero F32 si el runtime espera U8.

Nunca cuantizar posteriormente.

Escribir directamente:

```
F32

U8

I8
```

según el TensorSpec.

---

# Fase 16 — Batching

Mantener Rayon.

Pero:

Un hilo procesa una imagen completa.

No paralelizar dentro de una imagen.

Si el batch es muy grande:

usar chunks.

Reducir overhead de scheduling.

---

# Fase 17 — Doble Buffering

Mientras el runtime ejecuta inferencia:

la CPU debe preparar el siguiente tensor.

Nunca esperar a terminar la inferencia para comenzar el siguiente preprocessing.

---

# Fase 18 — Triple Buffering (opcional)

Permitir:

```
CPU

GPU

CPU

GPU
```

Siempre debe existir un buffer libre.

---

# Fase 19 — Backend de Resize

Crear abstracción:

```
ResizeBackend
```

Implementaciones:

```
fast_image_resize

OpenCV

Futuro interpolador propio
```

Todo el resto del pipeline debe ser independiente.

---

# Fase 20 — Interpolador propio (fase avanzada)

Objetivo final:

Eliminar completamente el buffer RGB escalado.

El algoritmo de interpolación debe escribir directamente en el tensor.

Flujo esperado:

```
RGB Original

↓

Interpolador Bilineal

↓

Tensor Final
```

Nunca existir:

```
RGB escalado
```

como buffer independiente.

---

# Fase 21 — Especialización por CPU

Implementar versiones específicas:

```
ARM64 + NEON

x86 + AVX2

x86 + AVX512
```

Seleccionadas automáticamente.

---

# Fase 22 — Benchmarks

Cada cambio debe medirse.

Crear benchmarks independientes para:

* resize
* pack
* tensor writer
* preprocessing completo
* batches 1
* batches 2
* batches 4
* batches 8
* batches 16

---

# Restricciones

No romper:

```
ModelPipeline
```

No modificar la API pública.

Toda optimización debe ser interna.

---

# Objetivo Final

El preprocessing debe comportarse como un **motor especializado de construcción de tensores**, no como una secuencia de transformaciones independientes.

Idealmente, el flujo efectivo debe aproximarse a:

```
Frame
   │
   ▼
Resize
   │
   ▼
TensorWriter
   ├── Padding
   ├── Layout
   ├── Conversión de tipo
   ├── Normalización
   ▼
TensorInput
```

Con una sola pasada sobre los datos tras el resize, reutilización completa de memoria, cero asignaciones durante el procesamiento normal y especialización por arquitectura cuando esté disponible. El objetivo es que el coste del preprocessing quede lo más cerca posible del coste intrínseco del resize, minimizando cualquier trabajo adicional.
