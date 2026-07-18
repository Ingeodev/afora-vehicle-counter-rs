[*]. Ajustar el tensor_spec que espera preprocessing y el que tiene internameinte runtime.
[ ] Crear la libreria de ffi para rga en los crates rga y sysrga
[ ] Crear las estrategias de preprocessing para CUDA, RKNN y CPU (Reutilizar logica existente de cpu ajustada)
[ ] Analizar la forma de evitar dependencias a librerias externas para eliminar problemas de compatibilidades.
[ ] Realizar prueba con compilacion a CUDA, probar con fallback en CPU para local y en Google Colab.
[ ] Volver a correr los benchmarks y analizar los tiempos.
[ ] Revisar el proceso de inferencia del modelo porque tarda mas de lo esperado.
[ ] Ajustar los pipelines para multihilo en las partes críticas.
[ ] Implementar el postprocessing para soporte de ppyoloe+
[ ] Implementar el postprocessing para rf-detr