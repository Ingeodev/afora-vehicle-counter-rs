#[cfg(all(feature="ppyoloe", feature="yolo11"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="ppyoloe", feature="rtdetr"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="ppyoloe", feature="rfdetr"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="yolo11", feature="rtdetr"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="yolo11", feature="rfdetr"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="rtdetr", feature="rfdetr"))]
compile_error!("Solo puede habilitarse un modelo.");

#[cfg(all(feature="rk3588", feature="cuda"))]
compile_error!("Solo puede habilitarse un backend.");

#[cfg(all(feature = "rk3588", feature = "rtdetr"))]
compile_error!("RT-DETR no está soportado en RK3588.");

#[cfg(all(feature = "rk3588", feature = "rfdetr"))]
compile_error!("RF-DETR no está soportado en RK3588.");