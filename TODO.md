1. Ajustar el tensor_spec que espera preprocessing y el que tiene internameinte runtime.
2. Crear las estrategias de preprocessing para CUDA, RKNN y CPU (Reutilizar logica existente de cpu ajustada)
3. Analizar la forma de evitar dependencias a librerias externas para eliminar problemas de compatibilidades.
4. Realizar prueba con compilacion a CUDA, probar con fallback en CPU para local y en Google Colab.
5. Volver a correr los benchmarks y analizar los tiempos.
6. Revisar el proceso de inferencia del modelo porque tarda mas de lo esperado.
7. Ajustar los pipelines para multihilo en las partes críticas.
8. Implementar el postprocessing para soporte de ppyoloe+
9. Implementar el postprocessing para rf-detr